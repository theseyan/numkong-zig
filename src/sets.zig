const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub fn hammingPacked(comptime dtype: types.DType, vectors: []const u8, queries_packed: []const u8, rows: usize, cols: usize, dimensions: usize, vector_stride_bytes: usize, out: []types.U32, out_stride_bytes: usize) void {
    switch (dtype) {
        .u1 => c.nk_hammings_packed_u1(@ptrCast(@alignCast(vectors.ptr)), queries_packed.ptr, out.ptr, rows, cols, dimensions, vector_stride_bytes, out_stride_bytes),
        else => @compileError("unsupported dtype for packed hamming"),
    }
}

pub fn hammingSymmetric(comptime dtype: types.DType, vectors: []const u8, vectors_count: usize, dimensions: usize, vector_stride_bytes: usize, out: []types.U32, out_stride_bytes: usize, row_start: usize, row_count: usize) void {
    switch (dtype) {
        .u1 => c.nk_hammings_symmetric_u1(@ptrCast(@alignCast(vectors.ptr)), vectors_count, dimensions, vector_stride_bytes, out.ptr, out_stride_bytes, row_start, row_count),
        else => @compileError("unsupported dtype for symmetric hamming"),
    }
}

pub fn jaccardPacked(comptime dtype: types.DType, vectors: []const u8, queries_packed: []const u8, rows: usize, cols: usize, dimensions: usize, vector_stride_bytes: usize, out: []types.F32, out_stride_bytes: usize) void {
    switch (dtype) {
        .u1 => c.nk_jaccards_packed_u1(@ptrCast(@alignCast(vectors.ptr)), queries_packed.ptr, out.ptr, rows, cols, dimensions, vector_stride_bytes, out_stride_bytes),
        else => @compileError("unsupported dtype for packed jaccard"),
    }
}

pub fn jaccardSymmetric(comptime dtype: types.DType, vectors: []const u8, vectors_count: usize, dimensions: usize, vector_stride_bytes: usize, out: []types.F32, out_stride_bytes: usize, row_start: usize, row_count: usize) void {
    switch (dtype) {
        .u1 => c.nk_jaccards_symmetric_u1(@ptrCast(@alignCast(vectors.ptr)), vectors_count, dimensions, vector_stride_bytes, out.ptr, out_stride_bytes, row_start, row_count),
        else => @compileError("unsupported dtype for symmetric jaccard"),
    }
}

test "sets packed and symmetric u1 wrappers" {
    const vectors = [_]types.U1x8{0xff};
    var packed_queries: [4096]u8 = undefined;
    const packed_size = try @import("dots.zig").packedSize(.u1, 1, 8);
    try @import("dots.zig").pack(.u1, std.mem.sliceAsBytes(vectors[0..]), 1, 8, 1, packed_queries[0..packed_size]);

    var hamming = [_]types.U32{1};
    hammingPacked(.u1, std.mem.sliceAsBytes(vectors[0..]), packed_queries[0..packed_size], 1, 1, 8, 1, &hamming, @sizeOf(types.U32));
    try std.testing.expectEqual(@as(types.U32, 0), hamming[0]);
    hamming[0] = 1;
    hammingSymmetric(.u1, std.mem.sliceAsBytes(vectors[0..]), 1, 8, 1, &hamming, @sizeOf(types.U32), 0, 1);
    try std.testing.expectEqual(@as(types.U32, 0), hamming[0]);

    var jaccard_out = [_]types.F32{1};
    jaccardPacked(.u1, std.mem.sliceAsBytes(vectors[0..]), packed_queries[0..packed_size], 1, 1, 8, 1, &jaccard_out, @sizeOf(types.F32));
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), jaccard_out[0], 1e-6);
    jaccard_out[0] = 1;
    jaccardSymmetric(.u1, std.mem.sliceAsBytes(vectors[0..]), 1, 8, 1, &jaccard_out, @sizeOf(types.F32), 0, 1);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), jaccard_out[0], 1e-6);
}
