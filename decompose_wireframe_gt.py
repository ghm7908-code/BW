import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

try:
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union
    HAS_SHAPELY = True
except ImportError:
    LineString = None
    polygonize = None
    unary_union = None
    HAS_SHAPELY = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose BuildingWorld/Building3D wireframe GT into roof/boundary/vertical/base parts."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Dataset root containing the wireframe directory.",
    )
    parser.add_argument(
        "--wireframe_subdir",
        default="wireframe",
        help="Subdirectory under dataset_root that stores GT wireframe OBJ files.",
    )
    parser.add_argument(
        "--output_subdir",
        default="decomp_gt_v1",
        help="Subdirectory under dataset_root used to store decomposition .npz files.",
    )
    parser.add_argument(
        "--summary_name",
        default="decomp_gt_v1_summary.jsonl",
        help="Summary JSONL filename written under dataset_root.",
    )
    parser.add_argument(
        "--failed_name",
        default="decomp_gt_v1_failed.txt",
        help="Failure log filename written under dataset_root.",
    )
    parser.add_argument(
        "--names_file",
        default="",
        help="Optional file listing sample ids to process.",
    )
    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Optional max number of samples to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing decomposition files.",
    )
    parser.add_argument(
        "--save_debug_obj",
        action="store_true",
        help="Write debug OBJ files for full/roof/base graphs.",
    )
    parser.add_argument(
        "--theta_vertical_deg",
        default=10.0,
        type=float,
        help="Maximum tilt angle from the z-axis for a line to be considered vertical.",
    )
    parser.add_argument(
        "--eps_xy_ratio",
        default=0.002,
        type=float,
        help="XY dedup tolerance ratio relative to XY diagonal length.",
    )
    parser.add_argument(
        "--eps_z_ratio",
        default=0.002,
        type=float,
        help="Z dedup tolerance ratio relative to total height range.",
    )
    parser.add_argument(
        "--min_vertical_span_ratio",
        default=0.01,
        type=float,
        help="Minimum z-span ratio relative to height range for a single edge to count as vertical.",
    )
    parser.add_argument(
        "--min_support_span_ratio",
        default=0.05,
        type=float,
        help="Minimum z-span ratio relative to height range for a connected vertical support chain.",
    )
    return parser.parse_args()


def parse_obj_index(token, num_vertices):
    raw = int(token.split("/")[0])
    if raw > 0:
        return raw - 1
    return num_vertices + raw


def load_wireframe_obj(obj_path):
    vertices = []
    edges = set()

    with open(obj_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "v":
                vertices.append([float(v) for v in parts[1:4]])
                continue

            if tag != "l":
                continue

            ids = [parse_obj_index(token, len(vertices)) for token in parts[1:]]
            if len(ids) < 2:
                continue

            for start_idx, end_idx in zip(ids[:-1], ids[1:]):
                if start_idx == end_idx:
                    continue
                edges.add(tuple(sorted((start_idx, end_idx))))

    vertices = np.asarray(vertices, dtype=np.float64)
    if len(edges) == 0:
        edges = np.zeros((0, 2), dtype=np.int32)
    else:
        edges = np.asarray(sorted(edges), dtype=np.int32)
    return vertices, edges


def compute_geometry_stats(vertices):
    xyz_min = vertices.min(axis=0)
    xyz_max = vertices.max(axis=0)
    extent = xyz_max - xyz_min
    diag_xy = float(np.linalg.norm(extent[:2]))
    height = float(extent[2])
    return {
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
        "extent": extent,
        "diag_xy": max(diag_xy, 1e-6),
        "height": max(height, 1e-6),
    }


def deduplicate_vertices(vertices, edges, eps_xy, eps_z):
    buckets = defaultdict(list)
    for index, vertex in enumerate(vertices):
        key = (
            int(round(vertex[0] / max(eps_xy, 1e-9))),
            int(round(vertex[1] / max(eps_xy, 1e-9))),
            int(round(vertex[2] / max(eps_z, 1e-9))),
        )
        buckets[key].append(index)

    new_vertices = []
    remap = np.full(len(vertices), -1, dtype=np.int32)
    for new_index, old_indices in enumerate(buckets.values()):
        merged = vertices[old_indices].mean(axis=0)
        new_vertices.append(merged)
        remap[old_indices] = new_index

    new_vertices = np.asarray(new_vertices, dtype=np.float64)

    new_edges = []
    for start_idx, end_idx in edges:
        mapped_start = remap[start_idx]
        mapped_end = remap[end_idx]
        if mapped_start == mapped_end:
            continue
        new_edges.append(tuple(sorted((int(mapped_start), int(mapped_end)))))

    if len(new_edges) == 0:
        new_edges = np.zeros((0, 2), dtype=np.int32)
    else:
        new_edges = np.asarray(sorted(set(new_edges)), dtype=np.int32)

    return new_vertices, new_edges, remap


def build_adjacency(num_vertices, edges):
    adjacency = [[] for _ in range(num_vertices)]
    for edge_index, (start_idx, end_idx) in enumerate(edges):
        adjacency[start_idx].append((end_idx, edge_index))
        adjacency[end_idx].append((start_idx, edge_index))
    return adjacency


def connected_components(num_vertices, adjacency, allowed_vertices=None, allowed_edge_mask=None):
    if allowed_vertices is None:
        allowed_vertices = np.ones(num_vertices, dtype=bool)
    if allowed_edge_mask is None:
        allowed_edge_mask = None

    visited = np.zeros(num_vertices, dtype=bool)
    components = []

    for start_vertex in range(num_vertices):
        if visited[start_vertex] or not allowed_vertices[start_vertex]:
            continue

        queue = deque([start_vertex])
        visited[start_vertex] = True
        component_vertices = []
        component_edges = set()

        while queue:
            vertex = queue.popleft()
            component_vertices.append(vertex)

            for neighbor, edge_index in adjacency[vertex]:
                if allowed_edge_mask is not None and not allowed_edge_mask[edge_index]:
                    continue
                if not allowed_vertices[neighbor]:
                    continue
                component_edges.add(edge_index)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(
            {
                "vertices": np.asarray(sorted(component_vertices), dtype=np.int32),
                "edges": np.asarray(sorted(component_edges), dtype=np.int32),
            }
        )

    return components


def detect_vertical_edges(vertices, edges, height, theta_vertical_deg, min_vertical_span_ratio):
    theta = math.radians(theta_vertical_deg)
    tan_theta = math.tan(theta)
    min_vertical_span = min_vertical_span_ratio * height

    mask = np.zeros(len(edges), dtype=bool)
    spans = np.zeros(len(edges), dtype=np.float64)

    for edge_index, (start_idx, end_idx) in enumerate(edges):
        p0 = vertices[start_idx]
        p1 = vertices[end_idx]
        delta = p1 - p0
        d_xy = float(np.linalg.norm(delta[:2]))
        d_z = float(abs(delta[2]))
        spans[edge_index] = d_z
        if d_z <= min_vertical_span:
            continue
        if d_xy / max(d_z, 1e-12) < tan_theta:
            mask[edge_index] = True

    return mask, spans


def extract_support_chains(vertices, edges, adjacency, vertical_edge_mask, height, min_support_span_ratio):
    support_components = connected_components(
        num_vertices=len(vertices),
        adjacency=adjacency,
        allowed_vertices=np.ones(len(vertices), dtype=bool),
        allowed_edge_mask=vertical_edge_mask,
    )

    min_support_span = min_support_span_ratio * height
    valid_supports = []

    for component in support_components:
        component_vertices = component["vertices"]
        if len(component_vertices) == 0:
            continue

        z_values = vertices[component_vertices, 2]
        top_local = int(np.argmax(z_values))
        bottom_local = int(np.argmin(z_values))
        top_vertex = int(component_vertices[top_local])
        bottom_vertex = int(component_vertices[bottom_local])
        z_span = float(z_values[top_local] - z_values[bottom_local])

        if z_span < min_support_span:
            continue

        valid_supports.append(
            {
                "vertices": component_vertices,
                "edges": component["edges"],
                "top_vertex": top_vertex,
                "bottom_vertex": bottom_vertex,
                "z_span": z_span,
            }
        )

    return valid_supports


def choose_best_support_per_top(valid_supports):
    best_by_top = {}
    for support in valid_supports:
        top_vertex = support["top_vertex"]
        previous = best_by_top.get(top_vertex)
        if previous is None or support["z_span"] > previous["z_span"]:
            best_by_top[top_vertex] = support
    return best_by_top


def extract_roof_graph(vertices, edges, adjacency, vertical_edge_mask, boundary_roof_vertices):
    non_vertical_mask = ~vertical_edge_mask
    components = connected_components(
        num_vertices=len(vertices),
        adjacency=adjacency,
        allowed_vertices=np.ones(len(vertices), dtype=bool),
        allowed_edge_mask=non_vertical_mask,
    )

    roof_vertex_ids = set()
    roof_edge_ids = set()

    for component in components:
        component_vertices = set(component["vertices"].tolist())
        if component_vertices & boundary_roof_vertices:
            roof_vertex_ids.update(component_vertices)
            roof_edge_ids.update(component["edges"].tolist())

    roof_vertex_ids = np.asarray(sorted(roof_vertex_ids), dtype=np.int32)
    roof_edge_ids = np.asarray(sorted(roof_edge_ids), dtype=np.int32)

    if len(roof_vertex_ids) == 0:
        return {
            "roof_vertices": np.zeros((0, 3), dtype=np.float64),
            "roof_edges": np.zeros((0, 2), dtype=np.int32),
            "roof_vertex_ids": roof_vertex_ids,
            "roof_edge_ids": roof_edge_ids,
            "full_to_roof_vertex": {},
        }

    full_to_roof_vertex = {int(vertex_id): local_id for local_id, vertex_id in enumerate(roof_vertex_ids.tolist())}
    roof_vertices = vertices[roof_vertex_ids]

    local_roof_edges = []
    for edge_id in roof_edge_ids:
        start_idx, end_idx = edges[edge_id]
        if int(start_idx) not in full_to_roof_vertex or int(end_idx) not in full_to_roof_vertex:
            continue
        local_roof_edges.append(
            [
                full_to_roof_vertex[int(start_idx)],
                full_to_roof_vertex[int(end_idx)],
            ]
        )

    if len(local_roof_edges) == 0:
        local_roof_edges = np.zeros((0, 2), dtype=np.int32)
    else:
        local_roof_edges = np.asarray(local_roof_edges, dtype=np.int32)

    return {
        "roof_vertices": roof_vertices,
        "roof_edges": local_roof_edges,
        "roof_vertex_ids": roof_vertex_ids,
        "roof_edge_ids": roof_edge_ids,
        "full_to_roof_vertex": full_to_roof_vertex,
    }


def classify_boundary_roof_edges_simple(roof_edges, roof_boundary_vertex_mask):
    if len(roof_edges) == 0:
        return np.zeros((0,), dtype=bool)
    return roof_boundary_vertex_mask[roof_edges[:, 0]] & roof_boundary_vertex_mask[roof_edges[:, 1]]


def quantize_point_2d(point_xy, eps_xy):
    return (
        int(round(float(point_xy[0]) / max(eps_xy, 1e-9))),
        int(round(float(point_xy[1]) / max(eps_xy, 1e-9))),
    )


def canonical_segment_key_2d(point_a, point_b, eps_xy):
    qa = quantize_point_2d(point_a, eps_xy)
    qb = quantize_point_2d(point_b, eps_xy)
    return tuple(sorted((qa, qb)))


def iter_linestring_segments(geometry):
    coords = list(geometry.coords)
    for idx in range(len(coords) - 1):
        yield np.asarray(coords[idx], dtype=np.float64), np.asarray(coords[idx + 1], dtype=np.float64)


def iter_polygon_boundary_segments(polygon):
    for point_a, point_b in iter_linestring_segments(polygon.exterior):
        yield point_a, point_b
    for ring in polygon.interiors:
        for point_a, point_b in iter_linestring_segments(ring):
            yield point_a, point_b


def classify_boundary_roof_edges_xy(roof_vertices, roof_edges, eps_xy, fallback_mask=None):
    if len(roof_edges) == 0 or len(roof_vertices) == 0:
        return np.zeros((0,), dtype=bool), "xy_boundary_ring_v2"

    if not HAS_SHAPELY:
        if fallback_mask is None:
            fallback_mask = np.zeros(len(roof_edges), dtype=bool)
        return fallback_mask.astype(bool), "simple_support_endpoint_rule_v1_fallback_no_shapely"

    edge_lines = []
    valid_lines = []
    for edge in roof_edges:
        point_a = roof_vertices[edge[0], :2]
        point_b = roof_vertices[edge[1], :2]
        if np.linalg.norm(point_a - point_b) <= max(eps_xy, 1e-8):
            edge_lines.append(None)
            continue
        line = LineString([tuple(point_a), tuple(point_b)])
        edge_lines.append(line)
        valid_lines.append(line)

    if not valid_lines:
        if fallback_mask is None:
            fallback_mask = np.zeros(len(roof_edges), dtype=bool)
        return fallback_mask.astype(bool), "simple_support_endpoint_rule_v1_fallback_no_valid_lines"

    merged = unary_union(valid_lines)
    polygons = list(polygonize(merged))
    if not polygons:
        if fallback_mask is None:
            fallback_mask = np.zeros(len(roof_edges), dtype=bool)
        return fallback_mask.astype(bool), "simple_support_endpoint_rule_v1_fallback_no_polygons"

    segment_usage = defaultdict(int)
    segment_geometry = {}
    for polygon in polygons:
        for point_a, point_b in iter_polygon_boundary_segments(polygon):
            key = canonical_segment_key_2d(point_a, point_b, eps_xy)
            segment_usage[key] += 1
            if key not in segment_geometry:
                segment_geometry[key] = LineString([tuple(point_a), tuple(point_b)])

    boundary_segments = [segment_geometry[key] for key, count in segment_usage.items() if count == 1]
    if not boundary_segments:
        if fallback_mask is None:
            fallback_mask = np.zeros(len(roof_edges), dtype=bool)
        return fallback_mask.astype(bool), "simple_support_endpoint_rule_v1_fallback_no_boundary_segments"

    boundary_union = unary_union(boundary_segments)
    boundary_mask = np.zeros(len(roof_edges), dtype=bool)

    for edge_index, line in enumerate(edge_lines):
        if line is None or line.length <= 1e-12:
            continue
        overlap_length = float(boundary_union.intersection(line).length)
        overlap_ratio = overlap_length / max(float(line.length), 1e-12)
        if overlap_ratio >= 0.95:
            boundary_mask[edge_index] = True
            continue
        if boundary_union.distance(line) <= max(eps_xy, 1e-6) and overlap_ratio >= 0.50:
            boundary_mask[edge_index] = True

    if not np.any(boundary_mask) and fallback_mask is not None:
        return fallback_mask.astype(bool), "simple_support_endpoint_rule_v1_fallback_empty_boundary"

    return boundary_mask, "xy_boundary_ring_v2"


def boundary_vertices_from_edges(num_roof_vertices, roof_edges, roof_boundary_edge_mask):
    boundary_vertex_mask = np.zeros(num_roof_vertices, dtype=bool)
    if len(roof_edges) == 0:
        return boundary_vertex_mask
    for start_idx, end_idx in roof_edges[roof_boundary_edge_mask]:
        boundary_vertex_mask[int(start_idx)] = True
        boundary_vertex_mask[int(end_idx)] = True
    return boundary_vertex_mask


def build_base_graph(vertices, roof_vertex_ids, roof_boundary_edge_mask, roof_edges, top_to_support, full_to_roof_vertex):
    base_vertex_ids = set()
    roof_vertex_ids_sorted = roof_vertex_ids.tolist()
    roof_has_support = np.zeros(len(roof_vertex_ids_sorted), dtype=bool)
    roof_base_points = np.full((len(roof_vertex_ids_sorted), 3), np.nan, dtype=np.float64)
    roof_base_heights = np.zeros(len(roof_vertex_ids_sorted), dtype=np.float64)
    pending_vertical_pairs = []
    roof_to_base = {}

    for full_vertex_id in roof_vertex_ids_sorted:
        if int(full_vertex_id) not in top_to_support:
            continue
        support = top_to_support[int(full_vertex_id)]
        base_vertex_ids.add(int(support["bottom_vertex"]))

    base_vertex_ids = np.asarray(sorted(base_vertex_ids), dtype=np.int32)
    full_to_base_vertex = {int(vertex_id): local_id for local_id, vertex_id in enumerate(base_vertex_ids.tolist())}
    base_vertices = vertices[base_vertex_ids] if len(base_vertex_ids) else np.zeros((0, 3), dtype=np.float64)

    for full_vertex_id in roof_vertex_ids_sorted:
        if int(full_vertex_id) not in top_to_support:
            continue
        roof_local = full_to_roof_vertex[int(full_vertex_id)]
        support = top_to_support[int(full_vertex_id)]
        base_full = int(support["bottom_vertex"])
        base_local = full_to_base_vertex[base_full]
        roof_has_support[roof_local] = True
        roof_base_points[roof_local] = vertices[base_full]
        roof_base_heights[roof_local] = float(vertices[full_vertex_id, 2] - vertices[base_full, 2])
        pending_vertical_pairs.append([roof_local, base_local])
        roof_to_base[int(roof_local)] = int(base_local)

    base_edges = set()
    for edge_idx, edge in enumerate(roof_edges):
        if not roof_boundary_edge_mask[edge_idx]:
            continue
        roof_u, roof_v = int(edge[0]), int(edge[1])
        if not roof_has_support[roof_u] or not roof_has_support[roof_v]:
            continue
        base_u = roof_to_base.get(roof_u)
        base_v = roof_to_base.get(roof_v)
        if base_u is None or base_v is None or base_u == base_v:
            continue
        base_edges.add(tuple(sorted((base_u, base_v))))

    if len(base_edges) == 0:
        base_edges = np.zeros((0, 2), dtype=np.int32)
    else:
        base_edges = np.asarray(sorted(base_edges), dtype=np.int32)

    vertical_pairs = pending_vertical_pairs
    if len(vertical_pairs) == 0:
        vertical_pairs = np.zeros((0, 2), dtype=np.int32)
    else:
        vertical_pairs = np.asarray(vertical_pairs, dtype=np.int32)

    return {
        "base_vertices": base_vertices,
        "base_edges": base_edges,
        "roof_has_support": roof_has_support,
        "roof_base_points": roof_base_points,
        "roof_base_heights": roof_base_heights,
        "vertical_pairs": vertical_pairs,
    }


def vertical_pairs_lookup(vertical_pairs, roof_local_vertex):
    for roof_local, base_local in vertical_pairs:
        if int(roof_local) == int(roof_local_vertex):
            return int(base_local)
    return None


def summarize_result(name, decomposition):
    roof_boundary_vertices = int(decomposition["roof_boundary_vertex_mask"].sum())
    roof_boundary_edges = int(decomposition["roof_boundary_edge_mask"].sum())
    support_count = int(decomposition["roof_has_support"].sum())
    summary = {
        "name": name,
        "full_vertices": int(len(decomposition["full_vertices"])),
        "full_edges": int(len(decomposition["full_edges"])),
        "roof_vertices": int(len(decomposition["roof_vertices"])),
        "roof_edges": int(len(decomposition["roof_edges"])),
        "boundary_roof_vertices": roof_boundary_vertices,
        "boundary_roof_edges": roof_boundary_edges,
        "supports": support_count,
        "base_vertices": int(len(decomposition["base_vertices"])),
        "base_edges": int(len(decomposition["base_edges"])),
        "boundary_rule": decomposition["metadata"]["boundary_rule"],
    }
    return summary


def save_wireframe_obj(vertices, edges, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write("v {:.6f} {:.6f} {:.6f}\n".format(vertex[0], vertex[1], vertex[2]))
        for start_idx, end_idx in edges:
            handle.write(f"l {start_idx + 1} {end_idx + 1}\n")


def decompose_single_wireframe(obj_path, args):
    vertices, edges = load_wireframe_obj(obj_path)
    if len(vertices) == 0:
        raise ValueError("empty wireframe vertices")

    stats = compute_geometry_stats(vertices)
    eps_xy = args.eps_xy_ratio * stats["diag_xy"]
    eps_z = args.eps_z_ratio * stats["height"]

    full_vertices, full_edges, _ = deduplicate_vertices(vertices, edges, eps_xy=eps_xy, eps_z=eps_z)
    if len(full_vertices) == 0:
        raise ValueError("all vertices collapsed during deduplication")

    full_stats = compute_geometry_stats(full_vertices)
    adjacency = build_adjacency(len(full_vertices), full_edges)

    vertical_edge_mask, _ = detect_vertical_edges(
        full_vertices,
        full_edges,
        height=full_stats["height"],
        theta_vertical_deg=args.theta_vertical_deg,
        min_vertical_span_ratio=args.min_vertical_span_ratio,
    )
    valid_supports = extract_support_chains(
        full_vertices,
        full_edges,
        adjacency,
        vertical_edge_mask=vertical_edge_mask,
        height=full_stats["height"],
        min_support_span_ratio=args.min_support_span_ratio,
    )
    top_to_support = choose_best_support_per_top(valid_supports)
    boundary_roof_vertices = set(top_to_support.keys())

    roof_graph = extract_roof_graph(
        full_vertices,
        full_edges,
        adjacency,
        vertical_edge_mask=vertical_edge_mask,
        boundary_roof_vertices=boundary_roof_vertices,
    )

    roof_vertices = roof_graph["roof_vertices"]
    roof_edges = roof_graph["roof_edges"]
    roof_vertex_ids = roof_graph["roof_vertex_ids"]
    full_to_roof_vertex = roof_graph["full_to_roof_vertex"]

    support_endpoint_vertex_mask = np.zeros(len(roof_vertices), dtype=bool)
    for full_vertex_id in boundary_roof_vertices:
        if int(full_vertex_id) in full_to_roof_vertex:
            support_endpoint_vertex_mask[full_to_roof_vertex[int(full_vertex_id)]] = True

    simple_boundary_edge_mask = classify_boundary_roof_edges_simple(
        roof_edges,
        support_endpoint_vertex_mask,
    )
    roof_boundary_edge_mask, boundary_rule = classify_boundary_roof_edges_xy(
        roof_vertices,
        roof_edges,
        eps_xy=eps_xy,
        fallback_mask=simple_boundary_edge_mask,
    )
    roof_boundary_vertex_mask = boundary_vertices_from_edges(
        len(roof_vertices),
        roof_edges,
        roof_boundary_edge_mask,
    )

    base_graph = build_base_graph(
        full_vertices,
        roof_vertex_ids=roof_vertex_ids,
        roof_boundary_edge_mask=roof_boundary_edge_mask,
        roof_edges=roof_edges,
        top_to_support=top_to_support,
        full_to_roof_vertex=full_to_roof_vertex,
    )

    metadata = {
        "source_obj": str(obj_path),
        "eps_xy": eps_xy,
        "eps_z": eps_z,
        "theta_vertical_deg": args.theta_vertical_deg,
        "min_vertical_span_ratio": args.min_vertical_span_ratio,
        "min_support_span_ratio": args.min_support_span_ratio,
        "boundary_rule": boundary_rule,
        "notes": (
            "Boundary roof edges are extracted from XY polygonized roof rings when possible, "
            "with fallback to the support-endpoint heuristic if polygonization fails."
        ),
    }

    return {
        "full_vertices": full_vertices.astype(np.float32),
        "full_edges": full_edges.astype(np.int32),
        "vertical_edge_mask": vertical_edge_mask.astype(np.uint8),
        "roof_vertices": roof_vertices.astype(np.float32),
        "roof_edges": roof_edges.astype(np.int32),
        "roof_boundary_vertex_mask": roof_boundary_vertex_mask.astype(np.uint8),
        "roof_boundary_edge_mask": roof_boundary_edge_mask.astype(np.uint8),
        "roof_has_support": base_graph["roof_has_support"].astype(np.uint8),
        "roof_base_points": base_graph["roof_base_points"].astype(np.float32),
        "roof_base_heights": base_graph["roof_base_heights"].astype(np.float32),
        "base_vertices": base_graph["base_vertices"].astype(np.float32),
        "base_edges": base_graph["base_edges"].astype(np.int32),
        "vertical_pairs": base_graph["vertical_pairs"].astype(np.int32),
        "roof_vertex_full_ids": roof_vertex_ids.astype(np.int32),
        "metadata": metadata,
    }


def write_npz(npz_path, decomposition):
    metadata_json = json.dumps(decomposition["metadata"])
    np.savez_compressed(
        npz_path,
        full_vertices=decomposition["full_vertices"],
        full_edges=decomposition["full_edges"],
        vertical_edge_mask=decomposition["vertical_edge_mask"],
        roof_vertices=decomposition["roof_vertices"],
        roof_edges=decomposition["roof_edges"],
        roof_boundary_vertex_mask=decomposition["roof_boundary_vertex_mask"],
        roof_boundary_edge_mask=decomposition["roof_boundary_edge_mask"],
        roof_has_support=decomposition["roof_has_support"],
        roof_base_points=decomposition["roof_base_points"],
        roof_base_heights=decomposition["roof_base_heights"],
        base_vertices=decomposition["base_vertices"],
        base_edges=decomposition["base_edges"],
        vertical_pairs=decomposition["vertical_pairs"],
        roof_vertex_full_ids=decomposition["roof_vertex_full_ids"],
        metadata_json=np.asarray(metadata_json),
    )


def resolve_sample_paths(wireframe_dir, names_file):
    obj_paths = sorted(wireframe_dir.glob("*.obj"))
    if names_file:
        with open(names_file, "r", encoding="utf-8") as handle:
            selected = {line.strip() for line in handle if line.strip()}
        obj_paths = [path for path in obj_paths if path.stem in selected]
    return obj_paths


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    wireframe_dir = dataset_root / args.wireframe_subdir
    output_dir = dataset_root / args.output_subdir
    summary_path = dataset_root / args.summary_name
    failed_path = dataset_root / args.failed_name
    debug_dir = output_dir / "debug_objs"

    if not wireframe_dir.exists():
        raise FileNotFoundError(f"Wireframe directory does not exist: {wireframe_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug_obj:
        debug_dir.mkdir(parents=True, exist_ok=True)

    obj_paths = resolve_sample_paths(wireframe_dir, args.names_file)
    if args.limit > 0:
        obj_paths = obj_paths[:args.limit]
    if not obj_paths:
        raise RuntimeError(f"No OBJ files found under {wireframe_dir}")

    summaries = []
    failures = []

    for index, obj_path in enumerate(obj_paths, start=1):
        name = obj_path.stem
        npz_path = output_dir / f"{name}.npz"
        if npz_path.exists() and not args.overwrite:
            print(f"Skip existing decomposition for {name}")
            continue

        try:
            decomposition = decompose_single_wireframe(obj_path, args)
            write_npz(npz_path, decomposition)
            summary = summarize_result(name, decomposition)
            summaries.append(summary)

            if args.save_debug_obj:
                sample_debug_dir = debug_dir / name
                save_wireframe_obj(
                    decomposition["full_vertices"],
                    decomposition["full_edges"],
                    sample_debug_dir / "full_cleaned.obj",
                )
                save_wireframe_obj(
                    decomposition["roof_vertices"],
                    decomposition["roof_edges"],
                    sample_debug_dir / "roof_graph.obj",
                )
                save_wireframe_obj(
                    decomposition["base_vertices"],
                    decomposition["base_edges"],
                    sample_debug_dir / "base_graph.obj",
                )

            if index % 100 == 0 or index == len(obj_paths):
                print(f"Processed {index}/{len(obj_paths)} samples")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"Failed {name}: {exc}")

    with open(summary_path, "w", encoding="utf-8") as handle:
        for item in summaries:
            handle.write(json.dumps(item) + "\n")

    with open(failed_path, "w", encoding="utf-8") as handle:
        for name, reason in failures:
            handle.write(f"{name}\t{reason}\n")

    print("\nDone")
    print(f"processed_successfully: {len(summaries)}")
    print(f"failed: {len(failures)}")
    print(f"output_dir: {output_dir}")
    print(f"summary_path: {summary_path}")
    print(f"failed_path: {failed_path}")


if __name__ == "__main__":
    main()
