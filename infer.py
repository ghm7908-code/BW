import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as filters
import skimage
import torch
from torch.utils.data import DataLoader

from datasets.data_utils import collate_fn, get_pixel_features
from datasets.test_outdoor_buildings import testOutdoorBuildingDataset
from models.corner_models import HeatCorner
from models.corner_models_3d import HeatCorner3d
from models.corner_to_edge import get_infer_edge_pairs
from models.edge_models import HeatEdge
from models.resnet import ResNetBackbone
from models.roof_structure_prior import RoofStructurePrior


def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def load_model_state(model, state_dict, strict=True):
    model_state = model.state_dict()
    has_module_prefix = next(iter(state_dict)).startswith('module.') if state_dict else False
    needs_module_prefix = next(iter(model_state)).startswith('module.') if model_state else False

    if has_module_prefix and not needs_module_prefix:
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    elif needs_module_prefix and not has_module_prefix:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)


def resolve_device(device_name):
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)

    if device.type != 'cuda':
        raise RuntimeError('BWFormer inference requires CUDA because the custom deformable attention op is CUDA-only.')
    return device


def save_wireframe(vertices, edges, wireframe_file):
    with open(wireframe_file, 'w') as f:
        for vertex in vertices:
            line = ' '.join(map(str, vertex))
            f.write('v ' + line + '\n')
        for edge in edges:
            edge = ' '.join(map(str, edge + 1))
            f.write('l ' + edge + '\n')



def corner_nms(preds, confs, image_size):
    data = np.zeros([image_size, image_size])
    neighborhood_size = 5
    threshold = 0

    for i in range(len(preds)):
        data[preds[i, 1], preds[i, 0]] = confs[i]

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = []
    for pred in filtered_preds:
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)

    return filtered_preds, new_confs



def main(dataset, ckpt_path, image_size, infer_times, data_path, test_list, pc_root, result_dir,
         num_workers, corner_thresh, device_name):
    device = resolve_device(device_name)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print('Load from ckpts of epoch {}'.format(ckpt['epoch']))
    ckpt_args = ckpt.get('args', argparse.Namespace())

    if pc_root is None:
        pc_root = os.path.join(data_path, 'xyz')
    os.makedirs(result_dir, exist_ok=True)

    det_path = None
    test_dataset = testOutdoorBuildingDataset(
        data_path,
        det_path,
        phase='test',
        image_size=image_size,
        rand_aug=False,
        inference=True,
        list_path=test_list or None,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    backbone = ResNetBackbone(pretrained=False).to(device)
    strides = backbone.strides
    num_channels = backbone.num_channels
    corner_model = HeatCorner(
        input_dim=128,
        hidden_dim=256,
        num_feature_levels=4,
        backbone_strides=strides,
        backbone_num_channels=num_channels,
    ).to(device)
    corner_model3d = HeatCorner3d(
        input_dim=128,
        hidden_dim=256,
        num_feature_levels=4,
        backbone_strides=strides,
        backbone_num_channels=num_channels,
    ).to(device)
    edge_model = HeatEdge(
        input_dim=128,
        hidden_dim=256,
        num_feature_levels=4,
        backbone_strides=strides,
        backbone_num_channels=num_channels,
    ).to(device)

    load_model_state(backbone, ckpt['backbone'])
    load_model_state(corner_model, ckpt['corner_model'], strict=False)
    load_model_state(corner_model3d, ckpt['corner_model3d'])
    load_model_state(edge_model, ckpt['edge_model'])

    roof_prior_model = RoofStructurePrior(hidden_dim=128).to(device)
    has_roof_prior = 'roof_prior_model' in ckpt
    if has_roof_prior:
        load_model_state(roof_prior_model, ckpt['roof_prior_model'])
        roof_prior_model.eval()
        print('Loaded roof prior model from checkpoint')
    else:
        print('Checkpoint has no roof_prior_model; running without roof feature fusion')

    backbone.eval()
    corner_model.eval()
    corner_model3d.eval()
    edge_model.eval()
    print('Loaded saved model from {}'.format(ckpt_path))

    pixels, pixel_features = get_pixel_features(image_size=image_size)
    pixel_features = pixel_features.to(device)

    for data_i, data in enumerate(test_dataloader):
        image = data['img'].to(device, non_blocking=True)
        img_name = data['name'][0]
        pc_file = os.path.join(pc_root, img_name + '.xyz')
        objpath = os.path.join(result_dir, img_name + '.obj')

        if not os.path.exists(pc_file):
            print('Skip {} because point cloud is missing: {}'.format(img_name, pc_file))
            save_wireframe([], np.empty((0, 2), dtype=np.int64), objpath)
            continue

        try:
            pc = np.loadtxt(pc_file, dtype=np.float64)
            point_cloud = pc[:, 0:3]

            with torch.no_grad():
                pred_corners, pred_confs, pos_edges, edge_confs, _ = get_results(
                    image,
                    backbone,
                    corner_model,
                    corner_model3d,
                    edge_model,
                    pixels,
                    pixel_features,
                    ckpt_args,
                    infer_times,
                    corner_thresh=corner_thresh,
                    image_size=image_size,
                    roof_prior_model=roof_prior_model if has_roof_prior else None,
                )

            if len(pred_corners) > 0:
                centroid = np.mean(point_cloud[:, 0:3], axis=0)
                point_cloud[:, 0:3] -= centroid
                max_distance = np.max(np.linalg.norm(point_cloud[:, 0:3], axis=1))
                if max_distance == 0:
                    max_distance = 1.0

                pred_corners = ((pred_corners - 127.5 * np.ones_like(pred_corners)) / 127.5) * max_distance + centroid
                pred_corners, pred_confs, pos_edges = postprocess_preds(pred_corners, pred_confs, pos_edges)
            else:
                pred_corners = np.empty((0, 3), dtype=np.float64)
                pred_confs = np.empty((0,), dtype=np.float64)
                pos_edges = np.empty((0, 2), dtype=np.int64)

            save_wireframe(pred_corners, pos_edges, objpath)
            print('Finished inference for sample {} ({}/{})'.format(img_name, data_i + 1, len(test_dataloader)))
        except Exception as e:
            print('Failed inference for {}: {}'.format(img_name, e))
            save_wireframe([], np.empty((0, 2), dtype=np.int64), objpath)



def get_results(image, backbone, corner_model, corner_model3d, edge_model, pixels, pixel_features,
                args, infer_times, corner_thresh=0.5, image_size=256, roof_prior_model=None):
    image_feats, feat_mask, all_image_feats = backbone(image)
    pixel_features = pixel_features.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)

    roof_features = None
    if roof_prior_model is not None:
        _, roof_features = roof_prior_model(all_image_feats)

    preds_s1 = corner_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats, roof_features)
    c_outputs = preds_s1
    maxnum = getattr(args, 'max_corner_num', 150)

    corners2d = prepare_corner_data(c_outputs, maxnum)
    corner_logits, corner_coord = corner_model3d(corners2d, image_feats, feat_mask, all_image_feats)

    c_outputs = corner_logits.sigmoid()
    corner_coord = torch.clip(corner_coord * 255, 0, 255)
    c_outputs_np = c_outputs[0].detach().cpu().numpy()
    c_coord_np = corner_coord[0].detach().cpu().numpy()

    pos_indices = np.where(c_outputs_np >= corner_thresh)
    coor_indices = np.array(pos_indices[0])
    pred_corners = c_coord_np[coor_indices, :]
    pred_confs = np.array(c_outputs_np[pos_indices])

    if pred_corners.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64), c_outputs_np

    sorted_indices = np.argsort(-pred_confs)
    pred_corners_sorted = pred_corners[sorted_indices]
    pred_confs_sorted = pred_confs[sorted_indices]
    newpred_corners = []
    newpred_conf = []
    for i in range(pred_confs_sorted.shape[0]):
        addyn = True
        for corner in newpred_corners:
            if calculate_distance(corner, pred_corners_sorted[i]) < 5:
                addyn = False
                break
        if addyn:
            newpred_corners.append(pred_corners_sorted[i])
            newpred_conf.append(pred_confs_sorted[i])
    pred_confs = np.array(newpred_conf)
    pred_corners = np.array(newpred_corners)

    if pred_corners.shape[0] < 2:
        if image_size != 256:
            pred_corners = pred_corners / (image_size / 256)
        return pred_corners, pred_confs, np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64), c_outputs_np

    pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(pred_corners, pred_confs)
    edge_coords = edge_coords.to(image.device, non_blocking=True)
    edge_mask = edge_mask.to(image.device, non_blocking=True)
    corner_nums = torch.tensor([len(pred_corners)], device=image.device)
    corner_to_edge_multiplier = getattr(args, 'corner_to_edge_multiplier', 3)
    max_candidates = torch.stack([corner_nums.max() * corner_to_edge_multiplier] * len(corner_nums), dim=0)

    all_pos_ids = set()
    all_edge_confs = {}
    gt_values = torch.full_like(edge_mask, 2, dtype=torch.long)

    for tt in range(infer_times):
        s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, s2_gt_values = edge_model(
            image_feats,
            feat_mask,
            pixel_features,
            edge_coords,
            edge_mask,
            gt_values,
            corner_nums,
            max_candidates,
            True,
        )

        num_total = s1_logits.shape[2]
        num_selected = selected_ids.shape[1]
        num_filtered = num_total - num_selected

        s2_preds_hb = s2_logits_hb.squeeze().softmax(0)
        s2_preds_np = s2_preds_hb[1, :].detach().cpu().numpy()
        selected_ids = selected_ids.squeeze().detach().cpu().numpy()

        if tt != infer_times - 1:
            pos_edge_ids = np.where(s2_preds_np >= 0.9)
            neg_edge_ids = np.where(s2_preds_np <= 0.01)
            for pos_id in pos_edge_ids[0]:
                actual_id = selected_ids[pos_id]
                if gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = s2_preds_np[pos_id]
                gt_values[0, actual_id] = 1
            for neg_id in neg_edge_ids[0]:
                actual_id = selected_ids[neg_id]
                if gt_values[0, actual_id] != 2:
                    continue
                gt_values[0, actual_id] = 0
            num_to_pred = (gt_values == 2).sum()
            if num_to_pred <= num_filtered:
                break
        else:
            pos_edge_ids = np.where(s2_preds_np >= 0.5)
            for pos_id in pos_edge_ids[0]:
                actual_id = selected_ids[pos_id]
                if s2_mask[0][pos_id] is True or gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = s2_preds_np[pos_id]

    pos_edge_ids = list(all_pos_ids)
    edge_confs = [all_edge_confs[idx] for idx in pos_edge_ids]
    pos_edges = edge_ids[pos_edge_ids].cpu().numpy() if len(pos_edge_ids) > 0 else np.empty((0, 2), dtype=np.int64)
    edge_confs = np.array(edge_confs)

    if image_size != 256:
        pred_corners = pred_corners / (image_size / 256)

    return pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np



def postprocess_preds(corners, confs, edges):
    corner_degrees = {}
    for edge_pair in edges:
        corner_degrees[edge_pair[0]] = corner_degrees.setdefault(edge_pair[0], 0) + 1
        corner_degrees[edge_pair[1]] = corner_degrees.setdefault(edge_pair[1], 0) + 1
    good_ids = [i for i in range(len(corners)) if i in corner_degrees]
    if len(good_ids) == len(corners):
        return corners, confs, edges
    if len(good_ids) == 0:
        return np.empty((0, corners.shape[1]), dtype=corners.dtype), np.empty((0,), dtype=confs.dtype), np.empty((0, 2), dtype=np.int64)

    good_corners = corners[good_ids]
    good_confs = confs[good_ids]
    id_mapping = {value: idx for idx, value in enumerate(good_ids)}
    new_edges = []
    for edge_pair in edges:
        new_pair = (id_mapping[edge_pair[0]], id_mapping[edge_pair[1]])
        new_edges.append(new_pair)
    new_edges = np.array(new_edges)
    return good_corners, good_confs, new_edges



def process_image(img, device=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = skimage.img_as_float(img)
    img = img.transpose((2, 0, 1))
    img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
    img = torch.Tensor(img)
    if device is not None:
        img = img.to(device)
    img = img.unsqueeze(0)
    return img



def plot_heatmap(results, filename):
    y, x = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))

    z = results[::-1, :]
    z = z[:-1, :-1]

    fig, ax = plt.subplots()

    c = ax.pcolormesh(y, x, z, cmap='RdBu', vmin=0, vmax=1)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    fig.savefig(filename)
    plt.close()



def convert_annot(annot):
    corners = np.array(list(annot.keys()))
    corners_mapping = {tuple(c): idx for idx, c in enumerate(corners)}
    edges = set()
    for corner, connections in annot.items():
        idx_c = corners_mapping[tuple(corner)]
        for other_c in connections:
            idx_other_c = corners_mapping[tuple(other_c)]
            if (idx_c, idx_other_c) not in edges and (idx_other_c, idx_c) not in edges:
                edges.add((idx_c, idx_other_c))
    edges = np.array(list(edges))
    gt_data = {
        'corners': corners,
        'edges': edges
    }
    return gt_data


def prepare_corner_data(c_outputs, max_corner_num):
    bs = c_outputs.shape[0]
    all_results = []

    for b_i in range(bs):
        output = c_outputs[b_i]
        results = process_each_corner_sample({'output': output}, max_corner_num)
        all_results.append(results)

    return torch.tensor(all_results).to(c_outputs.device)


def process_each_corner_sample(data, max_corner_num):
    output = data['output']

    preds = output.detach().cpu().numpy()
    neighbour_size = 5
    local_max_thresh = 0.01
    data_max = filters.maximum_filter(preds, neighbour_size)
    maxima = (preds == data_max)
    data_min = filters.minimum_filter(preds, neighbour_size)
    diff = ((data_max - data_min) > 0)
    maxima[diff == 0] = 0
    local_maximas = np.where((maxima > 0) & (preds > local_max_thresh))
    pred_corners = np.stack(local_maximas, axis=-1)[:, [1, 0]]
    if pred_corners.shape[0] > max_corner_num:
        indices = np.random.choice(pred_corners.shape[0], max_corner_num, replace=False)
        pred_corners = pred_corners[indices]
    if pred_corners.shape[0] < max_corner_num:
        num_padding = max_corner_num - pred_corners.shape[0]
        additional_corners = np.ones((num_padding, 2), dtype=pred_corners.dtype) * 255
        pred_corners = np.vstack((pred_corners, additional_corners))

    pred_corners = np.concatenate([pred_corners, pred_corners], axis=0)
    ind = np.lexsort(pred_corners.T)
    pred_corners = pred_corners[ind]
    return pred_corners



def get_args_parser():
    parser = argparse.ArgumentParser('Holistic edge attention transformer', add_help=False)
    parser.add_argument('--dataset', default='outdoor',
                        help='the dataset for experiments, outdoor/s3d_floorplan')
    parser.add_argument('--checkpoint_path', default='',
                        help='path to the checkpoints of the model')
    parser.add_argument('--data_path', default='/geogfs1/groups/hkurs/u3666068mgh/Tallin',
                        help='processed dataset root containing rgb/ and list files')
    parser.add_argument('--test_list', default='',
                        help='optional explicit test split file')
    parser.add_argument('--pc_root', default='',
                        help='directory containing raw .xyz point clouds for de-normalizing predictions')
    parser.add_argument('--result_dir', default='./results',
                        help='directory to write predicted .obj files')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--infer_times', default=3, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--corner_thresh', default=0.01, type=float)
    parser.add_argument('--device', default='auto',
                        help='device string such as auto or cuda:0')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HEAT inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(
        args.dataset,
        args.checkpoint_path,
        args.image_size,
        infer_times=args.infer_times,
        data_path=args.data_path,
        test_list=args.test_list,
        pc_root=args.pc_root or None,
        result_dir=args.result_dir,
        num_workers=args.num_workers,
        corner_thresh=args.corner_thresh,
        device_name=args.device,
    )
