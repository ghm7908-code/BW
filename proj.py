import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Project BuildingWorld wireframe + simulated point clouds into BWFormer training data.'
    )
    parser.add_argument(
        '--dataset_root',
        default='/geogfs1/groups/hkurs/u3666068mgh/Tallin',
        help='dataset root containing wireframe/ and simulated_pc/ subdirectories',
    )
    parser.add_argument('--pc_subdir', default='simulated_pc', help='point cloud subdirectory name')
    parser.add_argument('--wireframe_subdir', default='gt', help='wireframe/ground-truth OBJ subdirectory name')
    parser.add_argument('--rgb_subdir', default='rgb', help='output subdirectory for projected images')
    parser.add_argument('--annot_subdir', default='annot', help='output subdirectory for annotation npy files')
    parser.add_argument('--vis_subdir', default='vis', help='output subdirectory for visualization images')
    parser.add_argument('--image_size', default=256, type=int, help='target image size')
    parser.add_argument('--names_file', default='', help='optional file listing sample ids to process')
    parser.add_argument('--limit', default=0, type=int, help='optional cap on processed sample count')
    parser.add_argument(
        '--projection_mode',
        default='standard',
        choices=['standard', 'roof_prior_v1', 'roof_prior_v1b'],
        help='projection strategy: vanilla max-height map or facade-suppressed roof prior',
    )
    parser.add_argument(
        '--top_band_px',
        default=8.0,
        type=float,
        help='for roof-prior modes, keep only points within this normalized Z distance from each XY cell top surface',
    )
    parser.add_argument(
        '--blur_kernel',
        default=5,
        type=int,
        help='for roof-prior modes, odd Gaussian blur kernel used for density/edge smoothing',
    )
    parser.add_argument(
        '--height_gamma',
        default=0.55,
        type=float,
        help='for roof_prior_v1b, gamma used to lift darker low roofs while keeping height ordering',
    )
    parser.add_argument(
        '--valid_floor',
        default=0.18,
        type=float,
        help='for roof_prior_v1b, minimum normalized brightness assigned to valid roof pixels',
    )
    parser.add_argument('--overwrite', action='store_true', help='rebuild outputs even if they already exist')
    return parser.parse_args()


def parse_index(token):
    return int(token.split('/')[0]) - 1


def load_wireframe(wireframe_file):
    vertices = []
    edges = set()

    with open(wireframe_file, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == 'v':
                vertices.append([float(v) for v in parts[1:4]])
                continue

            if tag not in {'l', 'f'}:
                continue

            ids = [parse_index(token) for token in parts[1:]]
            if len(ids) < 2:
                continue

            for start, end in zip(ids[:-1], ids[1:]):
                if start == end:
                    continue
                edges.add(tuple(sorted((start, end))))

            if tag == 'f' and len(ids) > 2:
                edges.add(tuple(sorted((ids[-1], ids[0]))))

    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(sorted(edges), dtype=np.int32)
    return vertices, edges


def load_point_cloud(pc_file):
    point_cloud = np.loadtxt(pc_file, dtype=np.float64, usecols=(0, 1, 2))
    if point_cloud.ndim == 1:
        point_cloud = point_cloud.reshape(1, 3)
    return point_cloud


def normalize_geometry(point_cloud, wf_vertices, image_size):
    cloud_xyz = point_cloud[:, :3]
    centroid = np.mean(cloud_xyz, axis=0)
    centered = cloud_xyz - centroid
    max_distance = float(np.max(np.linalg.norm(centered, axis=1)))

    if max_distance <= 1e-8:
        raise ValueError('point cloud collapses to a single point')

    def transform(points):
        transformed = np.empty_like(points, dtype=np.float64)
        transformed = (points - centroid) / max_distance
        transformed = (transformed + 1.0) * 127.5
        transformed = np.clip(transformed, 0.0, image_size - 1.0)
        return transformed

    norm_point_cloud = transform(point_cloud[:, :3])
    norm_wf_vertices = transform(wf_vertices)
    return norm_point_cloud, norm_wf_vertices


def ensure_odd(kernel_size):
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def normalize_to_uint8(values, valid_mask=None, percentile=None, scale_max=None):
    image = np.zeros(values.shape, dtype=np.uint8)

    if valid_mask is None:
        valid_mask = np.ones(values.shape, dtype=bool)

    valid_values = values[valid_mask]
    if valid_values.size == 0:
        return image

    if scale_max is not None:
        denom = float(scale_max)
    elif percentile is not None:
        denom = float(np.percentile(valid_values, percentile))
    else:
        denom = float(valid_values.max())

    if denom <= 1e-8:
        return image

    scaled = np.clip(values / denom, 0.0, 1.0)
    image[valid_mask] = np.rint(scaled[valid_mask] * 255.0).astype(np.uint8)
    return image


def normalize_valid_range(values, valid_mask, lower_percentile=1.0, upper_percentile=99.0):
    normalized = np.zeros(values.shape, dtype=np.float32)
    valid_values = values[valid_mask]
    if valid_values.size == 0:
        return normalized

    lower = float(np.percentile(valid_values, lower_percentile))
    upper = float(np.percentile(valid_values, upper_percentile))

    if upper - lower <= 1e-8:
        lower = float(valid_values.min())
        upper = float(valid_values.max())
        if upper - lower <= 1e-8:
            normalized[valid_mask] = 1.0
            return normalized

    normalized = (values - lower) / (upper - lower)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    normalized[~valid_mask] = 0.0
    return normalized


def weighted_gaussian_blur(values, valid_mask, kernel_size):
    kernel_size = ensure_odd(kernel_size)
    weights = valid_mask.astype(np.float32)
    blurred_values = cv2.GaussianBlur(values.astype(np.float32) * weights, (kernel_size, kernel_size), 0)
    blurred_weights = cv2.GaussianBlur(weights, (kernel_size, kernel_size), 0)
    result = np.zeros(values.shape, dtype=np.float32)
    np.divide(blurred_values, np.maximum(blurred_weights, 1e-6), out=result, where=blurred_weights > 1e-6)
    result[~valid_mask] = 0.0
    return result


def compute_top_surface_maps(point_cloud, image_size, top_band_px):
    x_pixels = np.clip(np.floor(point_cloud[:, 0]).astype(np.int32), 0, image_size - 1)
    y_pixels = np.clip(np.floor(point_cloud[:, 1]).astype(np.int32), 0, image_size - 1)
    z_values = point_cloud[:, 2].astype(np.float32)
    flat_ids = y_pixels * image_size + x_pixels

    z_max_flat = np.full(image_size * image_size, -np.inf, dtype=np.float32)
    np.maximum.at(z_max_flat, flat_ids, z_values)

    local_top = z_max_flat[flat_ids]
    keep_mask = z_values >= (local_top - float(top_band_px))

    top_count_flat = np.zeros(image_size * image_size, dtype=np.int32)
    np.add.at(top_count_flat, flat_ids[keep_mask], 1)

    z_max_map = z_max_flat.reshape(image_size, image_size)
    top_count_map = top_count_flat.reshape(image_size, image_size)
    valid_mask = top_count_map > 0

    top_height_map = np.zeros((image_size, image_size), dtype=np.float32)
    top_height_map[valid_mask] = z_max_map[valid_mask]
    return top_height_map, top_count_map, valid_mask


def project_standard_height_map(point_cloud, image_size):
    x_pixels = np.clip(np.floor(point_cloud[:, 0]).astype(np.int32), 0, image_size - 1)
    y_pixels = np.clip(np.floor(point_cloud[:, 1]).astype(np.int32), 0, image_size - 1)
    z_values = np.clip(np.rint(point_cloud[:, 2]).astype(np.int32), 0, image_size - 1).astype(np.uint8)

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    depth_buffer = np.full((image_size, image_size), -np.inf, dtype=np.float64)

    for x_pixel, y_pixel, z_value, depth in zip(x_pixels, y_pixels, z_values, point_cloud[:, 2]):
        if depth > depth_buffer[y_pixel, x_pixel]:
            depth_buffer[y_pixel, x_pixel] = depth
            image[y_pixel, x_pixel] = (z_value, z_value, z_value)

    return image, image.copy()


def project_roof_prior_v1(point_cloud, image_size, top_band_px, blur_kernel):
    top_height_map, top_count_map, valid_mask = compute_top_surface_maps(point_cloud, image_size, top_band_px)

    blur_kernel = ensure_odd(blur_kernel)
    support_kernel = ensure_odd(max(blur_kernel * 2 + 1, 5))
    support_map = cv2.GaussianBlur(valid_mask.astype(np.float32), (support_kernel, support_kernel), 0)
    support_uint8 = normalize_to_uint8(support_map, valid_mask=valid_mask, percentile=99.0)
    support_weight = support_uint8.astype(np.float32) / 255.0
    roof_height_map = top_height_map * support_weight

    log_density_map = np.zeros((image_size, image_size), dtype=np.float32)
    log_density_map[valid_mask] = np.log1p(top_count_map[valid_mask].astype(np.float32))

    if blur_kernel > 1:
        density_smooth = cv2.GaussianBlur(log_density_map, (blur_kernel, blur_kernel), 0)
        height_smooth = cv2.GaussianBlur(roof_height_map, (blur_kernel, blur_kernel), 0)
    else:
        density_smooth = log_density_map
        height_smooth = roof_height_map

    height_uint8 = normalize_to_uint8(roof_height_map, valid_mask=valid_mask, scale_max=image_size - 1)
    density_uint8 = normalize_to_uint8(density_smooth, valid_mask=valid_mask, percentile=99.0)

    grad_x = cv2.Sobel(height_smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_smooth, cv2.CV_32F, 0, 1, ksize=3)
    edge_response = cv2.magnitude(grad_x, grad_y)
    edge_mask = cv2.dilate(valid_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    edge_response[~edge_mask] = 0.0
    edge_uint8 = normalize_to_uint8(edge_response, valid_mask=edge_mask, percentile=99.0)

    blue_channel = np.clip(0.7 * height_uint8.astype(np.float32) + 0.3 * support_uint8.astype(np.float32), 0, 255)
    green_channel = height_uint8.astype(np.float32)
    red_channel = np.clip(
        0.65 * height_uint8.astype(np.float32)
        + 0.2 * edge_uint8.astype(np.float32)
        + 0.15 * density_uint8.astype(np.float32),
        0,
        255,
    )
    projected_image = np.stack(
        [
            blue_channel.astype(np.uint8),
            green_channel.astype(np.uint8),
            red_channel.astype(np.uint8),
        ],
        axis=-1,
    )

    preview_gray = np.clip(
        0.75 * height_uint8.astype(np.float32) + 0.25 * support_uint8.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    preview_image = np.repeat(preview_gray[:, :, None], 3, axis=2)
    return projected_image, preview_image


def project_roof_prior_v1b(point_cloud, image_size, top_band_px, blur_kernel, height_gamma, valid_floor):
    top_height_map, top_count_map, valid_mask = compute_top_surface_maps(point_cloud, image_size, top_band_px)
    blur_kernel = ensure_odd(blur_kernel)

    support_kernel = ensure_odd(max(blur_kernel * 2 + 1, 5))
    support_map = cv2.GaussianBlur(valid_mask.astype(np.float32), (support_kernel, support_kernel), 0)
    support_norm = normalize_valid_range(support_map, valid_mask, lower_percentile=5.0, upper_percentile=99.5)

    height_norm = normalize_valid_range(top_height_map, valid_mask, lower_percentile=1.0, upper_percentile=99.0)
    height_lifted = np.zeros_like(height_norm, dtype=np.float32)
    if np.any(valid_mask):
        valid_floor = float(np.clip(valid_floor, 0.0, 0.95))
        height_gamma = float(max(height_gamma, 1e-3))
        height_lifted[valid_mask] = valid_floor + (1.0 - valid_floor) * np.power(
            height_norm[valid_mask],
            height_gamma,
        )

    log_density_map = np.zeros((image_size, image_size), dtype=np.float32)
    log_density_map[valid_mask] = np.log1p(top_count_map[valid_mask].astype(np.float32))
    density_smooth = weighted_gaussian_blur(log_density_map, valid_mask, blur_kernel)
    density_norm = normalize_valid_range(density_smooth, valid_mask, lower_percentile=1.0, upper_percentile=99.0)

    local_kernel = ensure_odd(max(blur_kernel * 4 + 1, 9))
    local_mean = weighted_gaussian_blur(top_height_map, valid_mask, local_kernel)
    local_relief = np.clip(top_height_map - local_mean, 0.0, None).astype(np.float32)
    local_relief_norm = normalize_valid_range(local_relief, valid_mask, lower_percentile=5.0, upper_percentile=99.0)

    height_smooth = weighted_gaussian_blur(height_lifted, valid_mask, blur_kernel)
    grad_x = cv2.Sobel(height_smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_smooth, cv2.CV_32F, 0, 1, ksize=3)
    edge_response = cv2.magnitude(grad_x, grad_y)
    edge_mask = cv2.dilate(valid_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    edge_response[~edge_mask] = 0.0
    edge_norm = normalize_valid_range(edge_response, edge_mask, lower_percentile=5.0, upper_percentile=99.0)

    blue_channel = np.clip(
        0.58 * height_lifted + 0.27 * support_norm + 0.15 * density_norm,
        0.0,
        1.0,
    )
    green_channel = np.clip(
        0.78 * height_lifted + 0.22 * support_norm,
        0.0,
        1.0,
    )
    red_channel = np.clip(
        0.50 * height_lifted + 0.20 * support_norm + 0.20 * edge_norm + 0.10 * local_relief_norm,
        0.0,
        1.0,
    )
    projected_image = np.stack(
        [
            np.rint(blue_channel * 255.0).astype(np.uint8),
            np.rint(green_channel * 255.0).astype(np.uint8),
            np.rint(red_channel * 255.0).astype(np.uint8),
        ],
        axis=-1,
    )

    preview_gray = np.clip(
        0.82 * height_lifted + 0.18 * support_norm,
        0.0,
        1.0,
    )
    preview_image = np.repeat(np.rint(preview_gray[:, :, None] * 255.0).astype(np.uint8), 3, axis=2)
    return projected_image, preview_image


def render_projection(point_cloud, image_size, projection_mode, top_band_px, blur_kernel, height_gamma, valid_floor):
    if projection_mode == 'standard':
        return project_standard_height_map(point_cloud, image_size)
    if projection_mode == 'roof_prior_v1':
        return project_roof_prior_v1(point_cloud, image_size, top_band_px, blur_kernel)
    if projection_mode == 'roof_prior_v1b':
        return project_roof_prior_v1b(point_cloud, image_size, top_band_px, blur_kernel, height_gamma, valid_floor)
    raise ValueError('Unsupported projection mode: {}'.format(projection_mode))


def build_vertex_connections(wf_vertices, wf_edges):
    vertex_connections = {tuple(vertex.tolist()): [] for vertex in wf_vertices}

    for start_idx, end_idx in wf_edges:
        start_vertex = tuple(wf_vertices[start_idx].tolist())
        end_vertex = wf_vertices[end_idx].copy()
        vertex_connections[start_vertex].append(end_vertex)

        end_vertex_key = tuple(wf_vertices[end_idx].tolist())
        start_vertex_value = wf_vertices[start_idx].copy()
        vertex_connections[end_vertex_key].append(start_vertex_value)

    return vertex_connections


def visualize_projection(image, vertex_connections, image_size):
    vis_image = image.copy()

    for vertex, neighbours in vertex_connections.items():
        x_coord = int(np.clip(round(vertex[0]), 0, image_size - 1))
        y_coord = int(np.clip(round(vertex[1]), 0, image_size - 1))
        cv2.circle(vis_image, (x_coord, y_coord), 2, (0, 0, 255), -1)

        for neighbour in neighbours:
            nx = int(np.clip(round(neighbour[0]), 0, image_size - 1))
            ny = int(np.clip(round(neighbour[1]), 0, image_size - 1))
            cv2.line(vis_image, (x_coord, y_coord), (nx, ny), (255, 0, 0), 1)

    return vis_image


def resolve_sample_names(pc_dir, wireframe_dir, names_file):
    pc_names = {path.stem for path in pc_dir.glob('*.xyz')}
    wireframe_names = {path.stem for path in wireframe_dir.glob('*.obj')}
    common_names = sorted(pc_names & wireframe_names)

    if names_file:
        with open(names_file, 'r', encoding='utf-8') as handle:
            selected_names = {line.strip() for line in handle if line.strip()}
        common_names = [name for name in common_names if name in selected_names]

    missing_pc = sorted(wireframe_names - pc_names)
    missing_wireframe = sorted(pc_names - wireframe_names)

    if missing_pc:
        print('Warning: {} wireframes have no matching .xyz file.'.format(len(missing_pc)))
    if missing_wireframe:
        print('Warning: {} point clouds have no matching .obj file.'.format(len(missing_wireframe)))

    return common_names, missing_pc, missing_wireframe


def process_one_sample(
    name,
    pc_dir,
    wireframe_dir,
    rgb_dir,
    annot_dir,
    vis_dir,
    image_size,
    projection_mode,
    top_band_px,
    blur_kernel,
    height_gamma,
    valid_floor,
    overwrite,
):
    rgb_path = rgb_dir / f'{name}.jpg'
    annot_path = annot_dir / f'{name}.npy'
    vis_path = vis_dir / f'{name}_vis.jpg'

    if (not overwrite) and rgb_path.exists() and annot_path.exists() and vis_path.exists():
        return True, None

    point_cloud = load_point_cloud(pc_dir / f'{name}.xyz')
    wf_vertices, wf_edges = load_wireframe(wireframe_dir / f'{name}.obj')

    if point_cloud.shape[0] == 0:
        raise ValueError('empty point cloud')
    if wf_vertices.shape[0] == 0:
        raise ValueError('wireframe has no vertices')

    norm_point_cloud, norm_wf_vertices = normalize_geometry(point_cloud, wf_vertices, image_size)
    vertex_connections = build_vertex_connections(norm_wf_vertices, wf_edges)
    projected_image, preview_image = render_projection(
        norm_point_cloud,
        image_size,
        projection_mode,
        top_band_px,
        blur_kernel,
        height_gamma,
        valid_floor,
    )
    vis_image = visualize_projection(preview_image, vertex_connections, image_size)

    np.save(annot_path, vertex_connections)
    cv2.imwrite(str(rgb_path), projected_image)
    cv2.imwrite(str(vis_path), vis_image)
    return True, None


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    pc_dir = dataset_root / args.pc_subdir
    wireframe_dir = dataset_root / args.wireframe_subdir
    rgb_dir = dataset_root / args.rgb_subdir
    annot_dir = dataset_root / args.annot_subdir
    vis_dir = dataset_root / args.vis_subdir

    for required_dir in [pc_dir, wireframe_dir]:
        if not required_dir.exists():
            raise FileNotFoundError('Required directory does not exist: {}'.format(required_dir))

    rgb_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_names, missing_pc, missing_wireframe = resolve_sample_names(pc_dir, wireframe_dir, args.names_file)
    if args.limit > 0:
        sample_names = sample_names[:args.limit]

    if not sample_names:
        raise RuntimeError('No matched .xyz/.obj pairs were found under {}'.format(dataset_root))

    processed_names = []
    failed_names = []

    for index, name in enumerate(sample_names, start=1):
        try:
            process_one_sample(
                name,
                pc_dir,
                wireframe_dir,
                rgb_dir,
                annot_dir,
                vis_dir,
                args.image_size,
                args.projection_mode,
                args.top_band_px,
                args.blur_kernel,
                args.height_gamma,
                args.valid_floor,
                args.overwrite,
            )
            processed_names.append(name)
        except Exception as exc:
            failed_names.append((name, str(exc)))

        if index % 100 == 0 or index == len(sample_names):
            print('Processed {}/{} samples'.format(index, len(sample_names)))

    all_list_path = dataset_root / 'all_list.txt'
    with open(all_list_path, 'w', encoding='utf-8') as handle:
        for name in processed_names:
            handle.write(name + '\n')

    missing_pc_path = dataset_root / 'missing_pointcloud.txt'
    with open(missing_pc_path, 'w', encoding='utf-8') as handle:
        for name in missing_pc:
            handle.write(name + '\n')

    missing_wireframe_path = dataset_root / 'missing_wireframe.txt'
    with open(missing_wireframe_path, 'w', encoding='utf-8') as handle:
        for name in missing_wireframe:
            handle.write(name + '\n')

    if failed_names:
        failed_log_path = dataset_root / 'failed_samples.txt'
        with open(failed_log_path, 'w', encoding='utf-8') as handle:
            for name, reason in failed_names:
                handle.write('{}\t{}\n'.format(name, reason))
        print('Finished with {} failures. See {}'.format(len(failed_names), failed_log_path))
    else:
        print('Finished without failures.')

    print('Projected images: {}'.format(rgb_dir))
    print('Annotations: {}'.format(annot_dir))
    print('Visualizations: {}'.format(vis_dir))
    print('Processed sample list: {}'.format(all_list_path))
    print('Missing point clouds: {}'.format(missing_pc_path))
    print('Missing wireframes: {}'.format(missing_wireframe_path))


if __name__ == '__main__':
    main()
