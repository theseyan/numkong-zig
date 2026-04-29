const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{UnsupportedDType};

pub fn angularPacked(comptime dtype: types.DType, a: []const u8, b_packed: []const u8, out: []u8, rows: usize, cols: usize, depth: usize, a_stride_bytes: usize, out_stride_bytes: usize) Error!void {
    switch (dtype) {
        .f32 => c.nk_angulars_packed_f32(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .f64 => c.nk_angulars_packed_f64(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .f16 => c.nk_angulars_packed_f16(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .bf16 => c.nk_angulars_packed_bf16(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .e4m3 => c.nk_angulars_packed_e4m3(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .e5m2 => c.nk_angulars_packed_e5m2(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .e2m3 => c.nk_angulars_packed_e2m3(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .e3m2 => c.nk_angulars_packed_e3m2(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .i8 => c.nk_angulars_packed_i8(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .u8 => c.nk_angulars_packed_u8(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .i4 => c.nk_angulars_packed_i4(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .u4 => c.nk_angulars_packed_u4(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        else => return Error.UnsupportedDType,
    }
}

pub fn euclideanPacked(comptime dtype: types.DType, a: []const u8, b_packed: []const u8, out: []u8, rows: usize, cols: usize, depth: usize, a_stride_bytes: usize, out_stride_bytes: usize) Error!void {
    switch (dtype) {
        .f32 => c.nk_euclideans_packed_f32(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .f64 => c.nk_euclideans_packed_f64(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .f16 => c.nk_euclideans_packed_f16(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .bf16 => c.nk_euclideans_packed_bf16(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .e4m3 => c.nk_euclideans_packed_e4m3(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .e5m2 => c.nk_euclideans_packed_e5m2(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .e2m3 => c.nk_euclideans_packed_e2m3(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .e3m2 => c.nk_euclideans_packed_e3m2(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .i8 => c.nk_euclideans_packed_i8(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .u8 => c.nk_euclideans_packed_u8(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .i4 => c.nk_euclideans_packed_i4(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        .u4 => c.nk_euclideans_packed_u4(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), rows, cols, depth, a_stride_bytes, out_stride_bytes),
        else => return Error.UnsupportedDType,
    }
}

pub fn angularSymmetric(comptime dtype: types.DType, vectors: []const u8, vectors_count: usize, depth: usize, stride_bytes: usize, out: []u8, out_stride_bytes: usize, row_start: usize, row_count: usize) Error!void {
    switch (dtype) {
        .f32 => c.nk_angulars_symmetric_f32(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .f64 => c.nk_angulars_symmetric_f64(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .f16 => c.nk_angulars_symmetric_f16(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .bf16 => c.nk_angulars_symmetric_bf16(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e4m3 => c.nk_angulars_symmetric_e4m3(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e5m2 => c.nk_angulars_symmetric_e5m2(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e2m3 => c.nk_angulars_symmetric_e2m3(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e3m2 => c.nk_angulars_symmetric_e3m2(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .i8 => c.nk_angulars_symmetric_i8(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .u8 => c.nk_angulars_symmetric_u8(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .i4 => c.nk_angulars_symmetric_i4(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .u4 => c.nk_angulars_symmetric_u4(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        else => return Error.UnsupportedDType,
    }
}

pub fn euclideanSymmetric(comptime dtype: types.DType, vectors: []const u8, vectors_count: usize, depth: usize, stride_bytes: usize, out: []u8, out_stride_bytes: usize, row_start: usize, row_count: usize) Error!void {
    switch (dtype) {
        .f32 => c.nk_euclideans_symmetric_f32(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .f64 => c.nk_euclideans_symmetric_f64(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .f16 => c.nk_euclideans_symmetric_f16(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .bf16 => c.nk_euclideans_symmetric_bf16(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e4m3 => c.nk_euclideans_symmetric_e4m3(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e5m2 => c.nk_euclideans_symmetric_e5m2(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e2m3 => c.nk_euclideans_symmetric_e2m3(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e3m2 => c.nk_euclideans_symmetric_e3m2(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .i8 => c.nk_euclideans_symmetric_i8(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .u8 => c.nk_euclideans_symmetric_u8(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .i4 => c.nk_euclideans_symmetric_i4(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .u4 => c.nk_euclideans_symmetric_u4(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        else => return Error.UnsupportedDType,
    }
}

test "spatials f32 packed and symmetric" {
    const vector = [_]types.F32{ 1, 2 };
    var packed_buffer: [4096]u8 = undefined;
    const packed_size = try @import("dots.zig").packedSize(.f32, 1, 2);
    try @import("dots.zig").pack(.f32, std.mem.sliceAsBytes(vector[0..]), 1, 2, 2 * @sizeOf(types.F32), packed_buffer[0..packed_size]);

    var packed_result = [_]types.F64{1};
    try angularPacked(.f32, std.mem.sliceAsBytes(vector[0..]), packed_buffer[0..packed_size], std.mem.sliceAsBytes(packed_result[0..]), 1, 1, 2, 2 * @sizeOf(types.F32), @sizeOf(types.F64));
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), packed_result[0], 1e-6);
    packed_result[0] = 1;
    try euclideanPacked(.f32, std.mem.sliceAsBytes(vector[0..]), packed_buffer[0..packed_size], std.mem.sliceAsBytes(packed_result[0..]), 1, 1, 2, 2 * @sizeOf(types.F32), @sizeOf(types.F64));
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), packed_result[0], 1e-6);

    var symmetric_result = [_]types.F64{1};
    try angularSymmetric(.f32, std.mem.sliceAsBytes(vector[0..]), 1, 2, 2 * @sizeOf(types.F32), std.mem.sliceAsBytes(symmetric_result[0..]), @sizeOf(types.F64), 0, 1);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), symmetric_result[0], 1e-6);
    symmetric_result[0] = 1;
    try euclideanSymmetric(.f32, std.mem.sliceAsBytes(vector[0..]), 1, 2, 2 * @sizeOf(types.F32), std.mem.sliceAsBytes(symmetric_result[0..]), @sizeOf(types.F64), 0, 1);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), symmetric_result[0], 1e-6);
}
