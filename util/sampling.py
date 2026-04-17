import copy
import random
import numpy as np
from tqdm import tqdm
from collections import deque
from sklearn.cluster import KMeans

import torch
import torch.nn as nn


def random_sampling(model, args):
    from train_selfsup import get_dataset

    # "sample_*" denotes the dataset for "*_sampling" function (val=True)
    sample_dataset2 = get_dataset(args, args.dataset2, args.data_folder2, val=True)
    sampling_nums = int(args.sampling_ratio * len(sample_dataset2))
    selected_indices = random.sample(range(len(sample_dataset2)), sampling_nums)

    print('Complete! {:d} number of indices sampled from {:s}.'.format(len(selected_indices), args.dataset2))

    return list(selected_indices)


# greedy selection algorithm for SimCore algorithm
def greedy(sim, args, sampling_nums=0):
    N, K = sim.shape
    queue_list = [deque(torch.argsort(sim[:,k], descending=True).numpy()) for k in range(K)]
    indices, tmp = set(), set()
    # for one iteration, each cluster picks one sample with the maximum utility function (i.e., maximum cosine similarity) 
    FLAG, threshold = 1, 0
    while len(indices) < sampling_nums:
        # for-loop according to the centroids
        func_value = 0
        for q, queue in enumerate(queue_list):
            while True:
                i = queue.popleft()
                if i not in indices: break
            tmp.update({i})
            if args.stop:
                func_value += sim[i, q].item() / K

        if FLAG == args.patience:
            threshold = func_value * args.stop_thresh
        FLAG += 1

        if args.stop and func_value < threshold:
            print('Stopped sampling because the rest are not similar to the target dataset.')
            break

        if len(indices) + len(tmp) > sampling_nums:
            print('Stopped sampling because of the limited budget.')
            tmp = set(np.random.choice(list(tmp), size=(sampling_nums - len(indices)), replace=False))
        indices.update(tmp)
        tmp = set()

    return indices


def simcore_sampling(model, args):
    from train_selfsup import get_dataset

    # "sample_*" denotes the dataset for "*_sampling" function (val=True)
    sample_dataset1 = get_dataset(args, args.dataset1, args.data_folder1, val=True)
    sample_dataset2 = get_dataset(args, args.dataset2, args.data_folder2, val=True)

    if args.stop:
        sampling_nums = 50 * len(sample_dataset1)
    else:
        sampling_nums = int(args.sampling_ratio * len(sample_dataset2))

    if sampling_nums == 0:
        assert args.from_ssl_official
        print('not sampling from open-set, and this is only for finetuning on X with a SSL official checkpoint')
        return []

    dataloader1 = torch.utils.data.DataLoader(sample_dataset1, batch_size=args.batch_size, shuffle=False, num_workers=1)
    dataloader2 = torch.utils.data.DataLoader(sample_dataset2, batch_size=args.batch_size, shuffle=False, num_workers=8)

    if args.method == 'mae':
        model.backbone.set_mask_ratio(mask_ratio=0)

    print('SimCore sampling starts!')
    model.eval()    
    with torch.no_grad():
        feats = []
        for images, _ in tqdm(dataloader1):
            img = images.cuda(non_blocking=True)
            
            z = model.forward_features(img)
            feats.append(nn.functional.normalize(z, dim=1).cpu())
        feats = torch.cat(feats, dim=0)

        # k-means clustering on normalized features
        if args.cluster_num < len(sample_dataset1):
            kmeans = KMeans(n_clusters=args.cluster_num, random_state=0).fit(feats.numpy())
            centroids = torch.tensor(kmeans.cluster_centers_).cuda()
        else:
            print('Using the centroids as each datapoint in {}.'.format(args.dataset1))
            centroids = feats.cuda()
        centroids = nn.functional.normalize(centroids, dim=1)

        del dataloader1
        print('Finding the closest {:d} samples from {:s} ...'.format(sampling_nums, args.dataset2))
        # sim is cosine similarity between centroids of target dataset(centroids) and features of openset(z)
        # sim = torch.tensor([], device=torch.device('cpu'))
        sim = []
        for idx, (images, _) in tqdm(enumerate(dataloader2)):
            img = images.cuda(non_blocking=True)
            z = model.forward_features(img)
            z = nn.functional.normalize(z, dim=1)
            
            sim.append(torch.mm(centroids, z.T).cpu())
        sim = torch.cat(sim, dim=1)
        print('Cosine similarity matrix is computed...')
        
        # get the solution of facility location problem in an iterative fashion
        selected_indices = greedy(sim.T.cpu(), args, sampling_nums=sampling_nums) # sim.shape == (# of openset, # of centroids)
        print('Complete! {:d} number of indices sampled from {:s}.'.format(len(selected_indices), args.dataset2))
        del dataloader2
        
    if args.method == 'mae':
        model.backbone.set_mask_ratio(mask_ratio=args.mask_ratio)
    return list(selected_indices)

##########################################
def craig_sampling(model, args):
    from train_selfsup import get_dataset
    import torch.nn.functional as F

    sample_dataset1 = get_dataset(args, args.dataset1, args.data_folder1, val=True)
    sample_dataset2 = get_dataset(args, args.dataset2, args.data_folder2, val=True)

    if args.stop:
        sampling_nums = 50 * len(sample_dataset1)
    else:
        sampling_nums = int(args.sampling_ratio * len(sample_dataset2))

    dataloader1 = torch.utils.data.DataLoader(sample_dataset1, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dataloader2 = torch.utils.data.DataLoader(sample_dataset2, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model.eval()

    def compute_ssl_gradient(images):
        """Trả về grad_vector (CPU Tensor) hoặc raise RuntimeError."""
        model.zero_grad()
        img = images.cuda(non_blocking=True)

        with torch.enable_grad():
            features = model.forward_features(img)
            pseudo_loss = features.norm(dim=1).mean()
            pseudo_loss.backward()

        # Thu thập gradient SAU khi thoát context (tránh side-effect)
        valid_grads = []
        for param in model.parameters():
            if param.grad is not None:
                valid_grads.append(param.grad.detach().cpu().view(-1))

        model.zero_grad()  # Dọn sau khi đã thu thập xong

        if len(valid_grads) == 0:
            raise RuntimeError(
                "Không tìm thấy gradient nào. Kiểm tra lại model có bị frozen hoàn toàn không."
            )

        # Chỉ lấy 2 layer cuối để tiết kiệm bộ nhớ
        grad_vector = torch.cat(valid_grads[-2:] if len(valid_grads) >= 2 else valid_grads)

        return grad_vector  # ← return nằm NGOÀI with block, đảm bảo luôn được thực thi

    # ===========================================
    # PHASE 1: TÍNH TARGET GRADIENTS CỦA DATASET1
    # ===========================================
    target_grads = []
    for images, _ in tqdm(dataloader1, desc="Phase 1: Target Gradients"):
        img_batch = images[0] if isinstance(images, list) else images
        grad_v = compute_ssl_gradient(img_batch)

        # Validation ngay tại đây để lỗi không bị ẩn đến torch.stack
        assert isinstance(grad_v, torch.Tensor), \
            f"compute_ssl_gradient trả về {type(grad_v)}, expected Tensor"
        target_grads.append(grad_v)

    if len(target_grads) == 0:
        raise RuntimeError("target_grads rỗng — dataloader1 không có dữ liệu?")

    target_grads = torch.stack(target_grads)  # [num_batches, grad_dim]

    # K-Means trên Gradient
    if args.cluster_num < len(target_grads):
        print('Clustering Target Gradients...')
        kmeans = KMeans(n_clusters=args.cluster_num, random_state=0).fit(target_grads.numpy())
        grad_centroids = torch.tensor(kmeans.cluster_centers_).cuda()
    else:
        grad_centroids = target_grads.cuda()

    grad_centroids = F.normalize(grad_centroids, p=2, dim=1)
    del dataloader1

    # ==========================================
    # PHASE 2: TÍNH POOL GRADIENTS CỦA DATASET2
    # ==========================================
    sim = []
    batch_sizes = []  # ← track actual batch size để map index chính xác

    for idx, (images, _) in tqdm(enumerate(dataloader2), desc="Phase 2: Pool Gradients"):
        img_batch = images[0] if isinstance(images, list) else images
        batch_sizes.append(len(img_batch))  # batch cuối có thể nhỏ hơn args.batch_size

        pool_grad = compute_ssl_gradient(img_batch).cuda()
        pool_grad = F.normalize(pool_grad.unsqueeze(0), p=2, dim=1)  # [1, grad_dim]

        similarity = torch.mm(grad_centroids, pool_grad.T).cpu()
        sim.append(similarity)

    sim = torch.cat(sim, dim=1)  # [cluster_num, num_batches]
    print('Cosine similarity matrix of GRADIENTS is computed...')

    # ==========================================
    # PHASE 3: GREEDY ALGO (FACILITY LOCATION)
    # ==========================================
    num_batches_to_select = (sampling_nums // args.batch_size) + 1
    selected_batch_indices = greedy(sim.T.cpu(), sampling_nums=num_batches_to_select)

    # Map batch index → image index, dùng batch_sizes thực tế để tránh out-of-range
    selected_indices = []
    cumulative = [0]
    for bs in batch_sizes:
        cumulative.append(cumulative[-1] + bs)

    for b_idx in selected_batch_indices:
        if b_idx >= len(batch_sizes):
            continue  # bỏ qua index không hợp lệ
        start_idx = cumulative[b_idx]
        end_idx = cumulative[b_idx + 1]
        selected_indices.extend(range(start_idx, end_idx))
        if len(selected_indices) >= sampling_nums:
            break

    selected_indices = selected_indices[:sampling_nums]
    print('Complete! {:d} number of indices sampled via CRAIG.'.format(len(selected_indices)))
    del dataloader2

    return list(selected_indices)
##########################################
def get_selected_indices(model, args):
    init_ckpt = copy.deepcopy(model.state_dict())
    if args.sampling_method == 'random':
        pass
    elif args.retrieval_ckpt is not None:
        print('pretrained retrieval model loaded from: {}'.format(args.retrieval_ckpt))
        model.load_state_dict(torch.load(args.retrieval_ckpt, weights_only=False)['model'])
    elif args.from_ssl_official:
        print('pretrained retrieval model loaded from SimCLR ImageNet-pretrained official checkpoint')
        assert args.method == 'simclr' and args.model == 'resnet50'
        if torch.cuda.device_count() > 1:
            model.backbone.module.load_ssl_official_weights()
        else:
            model.backbone.load_ssl_official_weights()
    else: 
        raise NotImplemented

    SAMPLING = {'random': random_sampling, 'simcore': simcore_sampling, 'craig': craig_sampling}
    selected_indices = SAMPLING[args.sampling_method](model, args)

    if args.stop:
        raw_epochs = copy.deepcopy(args.epochs)
        # re-calculate by the ratio to ImageNet length
        args.epochs = min(args.epochs, int(100 / (len(selected_indices) / 1281167)))
        if args.from_ssl_official: 
            args.epochs = int(args.epochs * 0.2) # fine-tuning the official pretrained model
        print('new epoch: {:d}'.format(args.epochs))
        
        if args.method == 'swav':
            args.freeze_prototypes = max(int((args.epochs * 10) / raw_epochs), 1)
            print('freeze prototypes: {}'.format(args.freeze_prototypes))
        if args.method == 'dino':
            if args.warm: args.warm_epochs = int((args.epochs * 100) / raw_epochs)
            args.freeze_last_layer = max(int((args.epochs * 10) / raw_epochs), 1)
            args.temp_warmup_epochs = int((args.epochs * 2000) / raw_epochs)
            model.temp_warmup_epochs = args.temp_warmup_epochs
            print('warm epochs: {}, freeze last layer: {}, temp warmup epochs: {}'.format(args.warm_epochs, args.freeze_last_layer, args.temp_warmup_epochs))
        if args.method == 'mae':
            if args.warm: args.warm_epochs = int((args.epochs * 100) / raw_epochs)
            print('warm epochs: {}'.format(args.warm_epochs))
    
    if args.sampling_times > 1: # multiple times of sampling coreset
        args.sampling_epochs = [int(args.epochs / args.sampling_times * (i+1)) for i in range(args.sampling_times-1)]
    else:
        args.sampling_epochs = []

    if not args.from_ssl_official:
        model.load_state_dict(init_ckpt) # initialize the model after openset sampling

    return selected_indices, model, args