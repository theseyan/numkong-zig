const std = @import("std");
const numkong = @import("numkong");
const zbench = @import("zbench");

const allocator = std.heap.c_allocator;
const types = numkong.types;

const dense_len = 4096;
const curved_dim = 64;
const geo_len = 2048;
const mesh_points = 128;
const matrix_rows = 16;
const matrix_cols = 16;
const matrix_depth = 256;
const sparse_len = 1024;
const u1_dimensions = 2048;
const u1_bytes = u1_dimensions / 8;
const maxsim_queries = 8;
const maxsim_docs = 16;

const Kernel = enum {
    types_dispatch_lookup,
    scalar_f32_sqrt,
    scalar_f32_fma,
    cast_f32_to_f16,
    cast_f16_to_f32,
    dot_f32,
    vdot_f32c,
    spatial_euclidean_f32,
    spatial_angular_f32,
    probability_kld_f32,
    probability_jsd_f32,
    each_scale_f32,
    each_sum_f32,
    each_fma_f32,
    each_sin_f32,
    reduce_moments_f32,
    reduce_minmax_f32,
    set_hamming_u8,
    set_jaccard_u32,
    set_hamming_u1,
    sparse_intersect_u32,
    sparse_dot_u32_f32,
    geospatial_haversine_f32,
    geospatial_vincenty_f32,
    curved_bilinear_f32,
    curved_mahalanobis_f32,
    mesh_rmsd_f32,
    mesh_kabsch_f32,
    dots_pack_f32,
    dots_packed_f32,
    dots_symmetric_f32,
    sets_hamming_packed_u1,
    sets_jaccard_symmetric_u1,
    spatials_angular_packed_f32,
    spatials_euclidean_symmetric_f32,
    maxsim_pack_f32,
    maxsim_compute_f32,
};

const BenchSpec = struct {
    name: []const u8,
    kernel: Kernel,
    batch: usize,
};

const specs = [_]BenchSpec{
    .{ .name = "types.findKernel dot/f32", .kernel = .types_dispatch_lookup, .batch = 4096 },
    .{ .name = "scalar.sqrt f32", .kernel = .scalar_f32_sqrt, .batch = 16384 },
    .{ .name = "scalar.fma f32", .kernel = .scalar_f32_fma, .batch = 16384 },
    .{ .name = "cast.slice f32->f16", .kernel = .cast_f32_to_f16, .batch = 64 },
    .{ .name = "cast.slice f16->f32", .kernel = .cast_f16_to_f32, .batch = 64 },
    .{ .name = "dot.dot f32", .kernel = .dot_f32, .batch = 64 },
    .{ .name = "dot.vdot f32c", .kernel = .vdot_f32c, .batch = 64 },
    .{ .name = "spatial.euclidean f32", .kernel = .spatial_euclidean_f32, .batch = 64 },
    .{ .name = "spatial.angular f32", .kernel = .spatial_angular_f32, .batch = 64 },
    .{ .name = "probability.kld f32", .kernel = .probability_kld_f32, .batch = 64 },
    .{ .name = "probability.jsd f32", .kernel = .probability_jsd_f32, .batch = 64 },
    .{ .name = "each.scale f32", .kernel = .each_scale_f32, .batch = 64 },
    .{ .name = "each.sum f32", .kernel = .each_sum_f32, .batch = 64 },
    .{ .name = "each.fma f32", .kernel = .each_fma_f32, .batch = 64 },
    .{ .name = "each.sin f32", .kernel = .each_sin_f32, .batch = 16 },
    .{ .name = "reduce.moments f32", .kernel = .reduce_moments_f32, .batch = 64 },
    .{ .name = "reduce.minmax f32", .kernel = .reduce_minmax_f32, .batch = 64 },
    .{ .name = "set.hamming u8", .kernel = .set_hamming_u8, .batch = 64 },
    .{ .name = "set.jaccard u32", .kernel = .set_jaccard_u32, .batch = 64 },
    .{ .name = "set.hamming u1", .kernel = .set_hamming_u1, .batch = 64 },
    .{ .name = "sparse.intersect u32", .kernel = .sparse_intersect_u32, .batch = 64 },
    .{ .name = "sparse.dot u32/f32", .kernel = .sparse_dot_u32_f32, .batch = 64 },
    .{ .name = "geospatial.haversine f32", .kernel = .geospatial_haversine_f32, .batch = 16 },
    .{ .name = "geospatial.vincenty f32", .kernel = .geospatial_vincenty_f32, .batch = 16 },
    .{ .name = "curved.bilinear f32", .kernel = .curved_bilinear_f32, .batch = 64 },
    .{ .name = "curved.mahalanobis f32", .kernel = .curved_mahalanobis_f32, .batch = 64 },
    .{ .name = "mesh.rmsd f32", .kernel = .mesh_rmsd_f32, .batch = 16 },
    .{ .name = "mesh.kabsch f32", .kernel = .mesh_kabsch_f32, .batch = 16 },
    .{ .name = "dots.pack f32", .kernel = .dots_pack_f32, .batch = 32 },
    .{ .name = "dots.computePacked f32", .kernel = .dots_packed_f32, .batch = 16 },
    .{ .name = "dots.symmetric f32", .kernel = .dots_symmetric_f32, .batch = 16 },
    .{ .name = "sets.hammingPacked u1", .kernel = .sets_hamming_packed_u1, .batch = 16 },
    .{ .name = "sets.jaccardSymmetric u1", .kernel = .sets_jaccard_symmetric_u1, .batch = 16 },
    .{ .name = "spatials.angularPacked f32", .kernel = .spatials_angular_packed_f32, .batch = 16 },
    .{ .name = "spatials.euclideanSymmetric f32", .kernel = .spatials_euclidean_symmetric_f32, .batch = 16 },
    .{ .name = "maxsim.pack f32", .kernel = .maxsim_pack_f32, .batch = 32 },
    .{ .name = "maxsim.compute f32", .kernel = .maxsim_compute_f32, .batch = 16 },
};

const Data = struct {
    f32_a: []types.F32,
    f32_b: []types.F32,
    f32_c: []types.F32,
    f32_out: []types.F32,
    f16_data: []types.F16,
    f32c_a: []types.F32C,
    f32c_b: []types.F32C,
    prob_a: []types.F32,
    prob_b: []types.F32,
    u8_a: []types.U8,
    u8_b: []types.U8,
    u32_a: []types.U32,
    u32_b: []types.U32,
    u1_a: []types.U1x8,
    u1_b: []types.U1x8,
    sparse_a: []types.U32,
    sparse_b: []types.U32,
    sparse_weights_a: []types.F32,
    sparse_weights_b: []types.F32,
    sparse_out: []types.U32,
    geo_a_lats: []types.F32,
    geo_a_lons: []types.F32,
    geo_b_lats: []types.F32,
    geo_b_lons: []types.F32,
    geo_out: []types.F32,
    curved_a: []types.F32,
    curved_b: []types.F32,
    curved_matrix: []types.F32,
    mesh_a: []types.F32,
    mesh_b: []types.F32,
    matrix_a: []types.F32,
    matrix_b: []types.F32,
    dots_packed: []u8,
    dots_out: []types.F64,
    dots_symmetric_out: []types.F64,
    u1_vectors: []types.U1x8,
    u1_queries: []types.U1x8,
    u1_packed: []u8,
    sets_out_u32: []types.U32,
    sets_out_f32: []types.F32,
    maxsim_queries_data: []types.F32,
    maxsim_docs_data: []types.F32,
    maxsim_query_packed: []u8,
    maxsim_doc_packed: []u8,

    fn init() !Data {
        var data: Data = undefined;
        data.f32_a = try allocator.alloc(types.F32, dense_len);
        data.f32_b = try allocator.alloc(types.F32, dense_len);
        data.f32_c = try allocator.alloc(types.F32, dense_len);
        data.f32_out = try allocator.alloc(types.F32, dense_len);
        data.f16_data = try allocator.alloc(types.F16, dense_len);
        data.f32c_a = try allocator.alloc(types.F32C, dense_len);
        data.f32c_b = try allocator.alloc(types.F32C, dense_len);
        data.prob_a = try allocator.alloc(types.F32, dense_len);
        data.prob_b = try allocator.alloc(types.F32, dense_len);
        data.u8_a = try allocator.alloc(types.U8, dense_len);
        data.u8_b = try allocator.alloc(types.U8, dense_len);
        data.u32_a = try allocator.alloc(types.U32, dense_len);
        data.u32_b = try allocator.alloc(types.U32, dense_len);
        data.u1_a = try allocator.alloc(types.U1x8, u1_bytes);
        data.u1_b = try allocator.alloc(types.U1x8, u1_bytes);
        data.sparse_a = try allocator.alloc(types.U32, sparse_len);
        data.sparse_b = try allocator.alloc(types.U32, sparse_len);
        data.sparse_weights_a = try allocator.alloc(types.F32, sparse_len);
        data.sparse_weights_b = try allocator.alloc(types.F32, sparse_len);
        data.sparse_out = try allocator.alloc(types.U32, sparse_len);
        data.geo_a_lats = try allocator.alloc(types.F32, geo_len);
        data.geo_a_lons = try allocator.alloc(types.F32, geo_len);
        data.geo_b_lats = try allocator.alloc(types.F32, geo_len);
        data.geo_b_lons = try allocator.alloc(types.F32, geo_len);
        data.geo_out = try allocator.alloc(types.F32, geo_len);
        data.curved_a = try allocator.alloc(types.F32, curved_dim);
        data.curved_b = try allocator.alloc(types.F32, curved_dim);
        data.curved_matrix = try allocator.alloc(types.F32, curved_dim * curved_dim);
        data.mesh_a = try allocator.alloc(types.F32, mesh_points * 3);
        data.mesh_b = try allocator.alloc(types.F32, mesh_points * 3);
        data.matrix_a = try allocator.alloc(types.F32, matrix_rows * matrix_depth);
        data.matrix_b = try allocator.alloc(types.F32, matrix_cols * matrix_depth);
        data.dots_out = try allocator.alloc(types.F64, matrix_rows * matrix_cols);
        data.dots_symmetric_out = try allocator.alloc(types.F64, matrix_rows * matrix_rows);
        data.u1_vectors = try allocator.alloc(types.U1x8, matrix_rows * u1_bytes);
        data.u1_queries = try allocator.alloc(types.U1x8, matrix_cols * u1_bytes);
        data.sets_out_u32 = try allocator.alloc(types.U32, matrix_rows * matrix_cols);
        data.sets_out_f32 = try allocator.alloc(types.F32, matrix_rows * matrix_rows);
        data.maxsim_queries_data = try allocator.alloc(types.F32, maxsim_queries * matrix_depth);
        data.maxsim_docs_data = try allocator.alloc(types.F32, maxsim_docs * matrix_depth);

        fillF32(data.f32_a, 0.001, 1);
        fillF32(data.f32_b, 0.002, 17);
        fillF32(data.f32_c, 0.003, 29);
        fillF32(data.prob_a, 1, 7);
        fillF32(data.prob_b, 1, 19);
        normalize(data.prob_a);
        normalize(data.prob_b);
        for (data.f32c_a, data.f32c_b, 0..) |*a, *b, i| {
            const real: f32 = @floatFromInt((i % 97) + 1);
            const imag: f32 = @floatFromInt(((i * 13) % 89) + 1);
            a.* = .{ .real = real * 0.001, .imag = imag * 0.001 };
            b.* = .{ .real = imag * 0.002, .imag = real * 0.002 };
        }
        numkong.cast.slice(std.mem.sliceAsBytes(data.f32_a), .f32, std.mem.sliceAsBytes(data.f16_data), .f16);

        for (data.u8_a, data.u8_b, 0..) |*a, *b, i| {
            a.* = @truncate((i * 37) + 0x5a);
            b.* = @truncate((i * 53) + 0xa5);
        }
        for (data.u32_a, data.u32_b, 0..) |*a, *b, i| {
            a.* = @intCast(i * 2);
            b.* = @intCast(i * 2);
        }
        fillU1(data.u1_a, 0x35);
        fillU1(data.u1_b, 0xa7);

        for (data.sparse_a, data.sparse_b, data.sparse_weights_a, data.sparse_weights_b, 0..) |*a, *b, *aw, *bw, i| {
            a.* = @intCast(i * 2);
            b.* = @intCast(i * 2);
            aw.* = @as(f32, @floatFromInt((i % 31) + 1)) * 0.01;
            bw.* = @as(f32, @floatFromInt((i % 37) + 1)) * 0.02;
        }

        fillGeo(data.geo_a_lats, data.geo_a_lons, 0);
        fillGeo(data.geo_b_lats, data.geo_b_lons, 11);
        fillF32(data.curved_a, 0.01, 3);
        fillF32(data.curved_b, 0.01, 5);
        fillIdentity(data.curved_matrix, curved_dim);
        fillF32(data.mesh_a, 0.01, 23);
        @memcpy(data.mesh_b, data.mesh_a);
        fillF32(data.matrix_a, 0.001, 31);
        fillF32(data.matrix_b, 0.001, 43);
        fillU1(data.u1_vectors, 0x11);
        fillU1(data.u1_queries, 0x73);
        fillF32(data.maxsim_queries_data, 0.001, 61);
        fillF32(data.maxsim_docs_data, 0.001, 71);

        const dots_packed_size = try numkong.dots.packedSize(.f32, matrix_cols, matrix_depth);
        data.dots_packed = try allocator.alloc(u8, dots_packed_size);
        try numkong.dots.pack(.f32, bytes(data.matrix_b), matrix_cols, matrix_depth, matrix_depth * @sizeOf(types.F32), data.dots_packed);

        const u1_packed_size = try numkong.dots.packedSize(.u1, matrix_cols, u1_dimensions);
        data.u1_packed = try allocator.alloc(u8, u1_packed_size);
        try numkong.dots.pack(.u1, bytes(data.u1_queries), matrix_cols, u1_dimensions, u1_bytes, data.u1_packed);

        const maxsim_query_size = try numkong.maxsim.packedSize(.f32, maxsim_queries, matrix_depth);
        const maxsim_doc_size = try numkong.maxsim.packedSize(.f32, maxsim_docs, matrix_depth);
        data.maxsim_query_packed = try allocator.alloc(u8, maxsim_query_size);
        data.maxsim_doc_packed = try allocator.alloc(u8, maxsim_doc_size);
        try numkong.maxsim.pack(.f32, bytes(data.maxsim_queries_data), maxsim_queries, matrix_depth, matrix_depth * @sizeOf(types.F32), data.maxsim_query_packed);
        try numkong.maxsim.pack(.f32, bytes(data.maxsim_docs_data), maxsim_docs, matrix_depth, matrix_depth * @sizeOf(types.F32), data.maxsim_doc_packed);

        return data;
    }
};

const Bench = struct {
    data: *Data,
    kernel: Kernel,
    batch: usize,

    pub fn run(self: *Bench, bench_allocator: std.mem.Allocator) void {
        _ = bench_allocator;
        var sink: f64 = 0;
        for (0..self.batch) |_| {
            sink += runKernel(self.data, self.kernel);
        }
        std.mem.doNotOptimizeAway(sink);
    }
};

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const writer = &stdout.interface;

    try writer.print(
        "NumKong binding benchmark suite\ncapabilities: detected=0x{x}, compiled=0x{x}, available=0x{x}\n\n",
        .{ types.capabilities(), types.capabilitiesCompiled(), types.capabilitiesAvailable() },
    );
    try writer.flush();

    var data = try Data.init();
    var bench = zbench.Benchmark.init(allocator, .{
        .time_budget_ns = 250_000_000,
        .max_iterations = 5000,
    });
    defer bench.deinit();

    var contexts: [specs.len]Bench = undefined;
    for (specs, 0..) |spec, index| {
        contexts[index] = .{ .data = &data, .kernel = spec.kernel, .batch = spec.batch };
        const name = try std.fmt.allocPrint(allocator, "{s} batch={d}", .{ spec.name, spec.batch });
        try bench.addParam(name, @as(*const Bench, &contexts[index]), .{});
    }

    try bench.run(init.io, std.Io.File.stdout());
}

fn runKernel(data: *Data, kernel: Kernel) f64 {
    return switch (kernel) {
        .types_dispatch_lookup => blk: {
            const lookup = types.findKernel(.dot, .f32, types.capabilitiesAvailable());
            break :blk @floatFromInt(lookup.capability);
        },
        .scalar_f32_sqrt => numkong.scalar.sqrt(.f32, 123.25),
        .scalar_f32_fma => numkong.scalar.fma(.f32, 1.25, 2.5, 3.75),
        .cast_f32_to_f16 => blk: {
            numkong.cast.slice(bytes(data.f32_a), .f32, bytes(data.f16_data), .f16);
            break :blk @floatFromInt(data.f16_data[0]);
        },
        .cast_f16_to_f32 => blk: {
            numkong.cast.slice(bytes(data.f16_data), .f16, bytes(data.f32_out), .f32);
            break :blk data.f32_out[0];
        },
        .dot_f32 => numkong.dot.dot(.f32, bytes(data.f32_a), bytes(data.f32_b), data.f32_a.len) catch unreachable,
        .vdot_f32c => blk: {
            const result = numkong.dot.vdot(.f32c, bytes(data.f32c_a), bytes(data.f32c_b), data.f32c_a.len) catch unreachable;
            break :blk result.real + result.imag;
        },
        .spatial_euclidean_f32 => numkong.spatial.euclidean(.f32, bytes(data.f32_a), bytes(data.f32_b), data.f32_a.len) catch unreachable,
        .spatial_angular_f32 => numkong.spatial.angular(.f32, bytes(data.f32_a), bytes(data.f32_b), data.f32_a.len) catch unreachable,
        .probability_kld_f32 => numkong.probability.kld(.f32, bytes(data.prob_a), bytes(data.prob_b), data.prob_a.len) catch unreachable,
        .probability_jsd_f32 => numkong.probability.jsd(.f32, bytes(data.prob_a), bytes(data.prob_b), data.prob_a.len) catch unreachable,
        .each_scale_f32 => blk: {
            numkong.each.scale(.f32, bytes(data.f32_a), 1.25, 0.5, bytes(data.f32_out)) catch unreachable;
            break :blk data.f32_out[0];
        },
        .each_sum_f32 => blk: {
            numkong.each.sum(.f32, bytes(data.f32_a), bytes(data.f32_b), bytes(data.f32_out)) catch unreachable;
            break :blk data.f32_out[0];
        },
        .each_fma_f32 => blk: {
            numkong.each.fma(.f32, bytes(data.f32_a), bytes(data.f32_b), bytes(data.f32_c), 0.75, 0.25, bytes(data.f32_out)) catch unreachable;
            break :blk data.f32_out[0];
        },
        .each_sin_f32 => blk: {
            numkong.each.sin(.f32, bytes(data.f32_a), bytes(data.f32_out)) catch unreachable;
            break :blk data.f32_out[0];
        },
        .reduce_moments_f32 => blk: {
            const result = numkong.reduce.moments(.f32, bytes(data.f32_a), data.f32_a.len) catch unreachable;
            break :blk result.sum + result.sumsq;
        },
        .reduce_minmax_f32 => blk: {
            const result = numkong.reduce.minmax(.f32, bytes(data.f32_a), data.f32_a.len) catch unreachable;
            break :blk result.min + result.max + @as(f64, @floatFromInt(result.min_index + result.max_index));
        },
        .set_hamming_u8 => @floatFromInt(numkong.set.hamming(.u8, bytes(data.u8_a), bytes(data.u8_b), data.u8_a.len) catch unreachable),
        .set_jaccard_u32 => numkong.set.jaccard(.u32, bytes(data.u32_a), bytes(data.u32_b), data.u32_a.len) catch unreachable,
        .set_hamming_u1 => @floatFromInt(numkong.set.hamming(.u1, bytes(data.u1_a), bytes(data.u1_b), u1_dimensions) catch unreachable),
        .sparse_intersect_u32 => @floatFromInt(numkong.sparse.intersect(.u32, data.sparse_a, data.sparse_b, data.sparse_out)),
        .sparse_dot_u32_f32 => numkong.sparse.dot(.u32, .f32, data.sparse_a, data.sparse_b, data.sparse_weights_a, data.sparse_weights_b),
        .geospatial_haversine_f32 => blk: {
            numkong.geospatial.haversine(.f32, bytes(data.geo_a_lats), bytes(data.geo_a_lons), bytes(data.geo_b_lats), bytes(data.geo_b_lons), bytes(data.geo_out), geo_len) catch unreachable;
            break :blk data.geo_out[0];
        },
        .geospatial_vincenty_f32 => blk: {
            numkong.geospatial.vincenty(.f32, bytes(data.geo_a_lats), bytes(data.geo_a_lons), bytes(data.geo_b_lats), bytes(data.geo_b_lons), bytes(data.geo_out), geo_len) catch unreachable;
            break :blk data.geo_out[0];
        },
        .curved_bilinear_f32 => numkong.curved.bilinear(.f32, bytes(data.curved_a), bytes(data.curved_b), bytes(data.curved_matrix), curved_dim) catch unreachable,
        .curved_mahalanobis_f32 => numkong.curved.mahalanobis(.f32, bytes(data.curved_a), bytes(data.curved_b), bytes(data.curved_matrix), curved_dim) catch unreachable,
        .mesh_rmsd_f32 => (numkong.mesh.rmsd(.f32, bytes(data.mesh_a), bytes(data.mesh_b), mesh_points) catch unreachable).rmsd,
        .mesh_kabsch_f32 => (numkong.mesh.kabsch(.f32, bytes(data.mesh_a), bytes(data.mesh_b), mesh_points) catch unreachable).rmsd,
        .dots_pack_f32 => blk: {
            numkong.dots.pack(.f32, bytes(data.matrix_b), matrix_cols, matrix_depth, matrix_depth * @sizeOf(types.F32), data.dots_packed) catch unreachable;
            break :blk @floatFromInt(data.dots_packed[0]);
        },
        .dots_packed_f32 => blk: {
            numkong.dots.computePacked(.f32, bytes(data.matrix_a), data.dots_packed, matrix_rows, matrix_cols, matrix_depth, matrix_depth * @sizeOf(types.F32), bytes(data.dots_out), matrix_cols * @sizeOf(types.F64)) catch unreachable;
            break :blk data.dots_out[0];
        },
        .dots_symmetric_f32 => blk: {
            numkong.dots.symmetric(.f32, bytes(data.matrix_a), matrix_rows, matrix_depth, matrix_depth * @sizeOf(types.F32), bytes(data.dots_symmetric_out), matrix_rows * @sizeOf(types.F64), 0, matrix_rows) catch unreachable;
            break :blk data.dots_symmetric_out[0];
        },
        .sets_hamming_packed_u1 => blk: {
            numkong.sets.hammingPacked(.u1, bytes(data.u1_vectors), data.u1_packed, matrix_rows, matrix_cols, u1_dimensions, u1_bytes, data.sets_out_u32, matrix_cols * @sizeOf(types.U32));
            break :blk @floatFromInt(data.sets_out_u32[0]);
        },
        .sets_jaccard_symmetric_u1 => blk: {
            numkong.sets.jaccardSymmetric(.u1, bytes(data.u1_vectors), matrix_rows, u1_dimensions, u1_bytes, data.sets_out_f32, matrix_rows * @sizeOf(types.F32), 0, matrix_rows);
            break :blk data.sets_out_f32[0];
        },
        .spatials_angular_packed_f32 => blk: {
            numkong.spatials.angularPacked(.f32, bytes(data.matrix_a), data.dots_packed, bytes(data.dots_out), matrix_rows, matrix_cols, matrix_depth, matrix_depth * @sizeOf(types.F32), matrix_cols * @sizeOf(types.F64)) catch unreachable;
            break :blk data.dots_out[0];
        },
        .spatials_euclidean_symmetric_f32 => blk: {
            numkong.spatials.euclideanSymmetric(.f32, bytes(data.matrix_a), matrix_rows, matrix_depth, matrix_depth * @sizeOf(types.F32), bytes(data.dots_symmetric_out), matrix_rows * @sizeOf(types.F64), 0, matrix_rows) catch unreachable;
            break :blk data.dots_symmetric_out[0];
        },
        .maxsim_pack_f32 => blk: {
            numkong.maxsim.pack(.f32, bytes(data.maxsim_docs_data), maxsim_docs, matrix_depth, matrix_depth * @sizeOf(types.F32), data.maxsim_doc_packed) catch unreachable;
            break :blk @floatFromInt(data.maxsim_doc_packed[0]);
        },
        .maxsim_compute_f32 => numkong.maxsim.compute(.f32, data.maxsim_query_packed, data.maxsim_doc_packed, maxsim_queries, maxsim_docs, matrix_depth) catch unreachable,
    };
}

fn bytes(slice: anytype) []u8 {
    return std.mem.sliceAsBytes(slice);
}

fn fillF32(values: []types.F32, scale: types.F32, seed: usize) void {
    for (values, 0..) |*value, i| {
        value.* = @as(types.F32, @floatFromInt(((i + seed) * 17) % 251 + 1)) * scale;
    }
}

fn normalize(values: []types.F32) void {
    var sum: types.F32 = 0;
    for (values) |value| sum += value;
    for (values) |*value| value.* /= sum;
}

fn fillU1(values: []types.U1x8, seed: u8) void {
    for (values, 0..) |*value, i| {
        value.* = @truncate((i * 37) ^ seed);
    }
}

fn fillGeo(lats: []types.F32, lons: []types.F32, seed: usize) void {
    for (lats, lons, 0..) |*lat, *lon, i| {
        lat.* = (@as(types.F32, @floatFromInt(((i + seed) % 180))) - 90) * 0.01;
        lon.* = (@as(types.F32, @floatFromInt(((i * 7 + seed) % 360))) - 180) * 0.01;
    }
}

fn fillIdentity(matrix: []types.F32, dimension: usize) void {
    for (matrix, 0..) |*value, index| {
        const row = index / dimension;
        const col = index % dimension;
        value.* = if (row == col) 1 else 0;
    }
}
