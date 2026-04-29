const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{UnsupportedDType};

pub fn packedSize(comptime dtype: types.DType, width: usize, depth: usize) Error!usize {
    return switch (dtype) {
        .bf16 => c.nk_dots_packed_size_bf16(width, depth),
        .f16 => c.nk_dots_packed_size_f16(width, depth),
        .e4m3 => c.nk_dots_packed_size_e4m3(width, depth),
        .e5m2 => c.nk_dots_packed_size_e5m2(width, depth),
        .e2m3 => c.nk_dots_packed_size_e2m3(width, depth),
        .e3m2 => c.nk_dots_packed_size_e3m2(width, depth),
        .f32 => c.nk_dots_packed_size_f32(width, depth),
        .f64 => c.nk_dots_packed_size_f64(width, depth),
        .i8 => c.nk_dots_packed_size_i8(width, depth),
        .u8 => c.nk_dots_packed_size_u8(width, depth),
        .i4 => c.nk_dots_packed_size_i4(width, depth),
        .u4 => c.nk_dots_packed_size_u4(width, depth),
        .u1 => c.nk_dots_packed_size_u1(width, depth),
        else => Error.UnsupportedDType,
    };
}

pub fn pack(comptime dtype: types.DType, b: []const u8, width: usize, depth: usize, row_stride_bytes: usize, out: []u8) Error!void {
    std.debug.assert(out.len >= try packedSize(dtype, width, depth));
    switch (dtype) {
        .bf16 => c.nk_dots_pack_bf16(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .f16 => c.nk_dots_pack_f16(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .e4m3 => c.nk_dots_pack_e4m3(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .e5m2 => c.nk_dots_pack_e5m2(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .e2m3 => c.nk_dots_pack_e2m3(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .e3m2 => c.nk_dots_pack_e3m2(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .f32 => c.nk_dots_pack_f32(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .f64 => c.nk_dots_pack_f64(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .i8 => c.nk_dots_pack_i8(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .u8 => c.nk_dots_pack_u8(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .i4 => c.nk_dots_pack_i4(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .u4 => c.nk_dots_pack_u4(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        .u1 => c.nk_dots_pack_u1(@ptrCast(@alignCast(b.ptr)), width, depth, row_stride_bytes, out.ptr),
        else => return Error.UnsupportedDType,
    }
}

pub fn computePacked(comptime dtype: types.DType, a: []const u8, b_packed: []const u8, height: usize, width: usize, depth: usize, a_stride_bytes: usize, out: []u8, out_stride_bytes: usize) Error!void {
    switch (dtype) {
        .bf16 => c.nk_dots_packed_bf16(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .f16 => c.nk_dots_packed_f16(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .e4m3 => c.nk_dots_packed_e4m3(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .e5m2 => c.nk_dots_packed_e5m2(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .e2m3 => c.nk_dots_packed_e2m3(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .e3m2 => c.nk_dots_packed_e3m2(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .f32 => c.nk_dots_packed_f32(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .f64 => c.nk_dots_packed_f64(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .i8 => c.nk_dots_packed_i8(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .u8 => c.nk_dots_packed_u8(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .i4 => c.nk_dots_packed_i4(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .u4 => c.nk_dots_packed_u4(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        .u1 => c.nk_dots_packed_u1(@ptrCast(@alignCast(a.ptr)), b_packed.ptr, @ptrCast(@alignCast(out.ptr)), height, width, depth, a_stride_bytes, out_stride_bytes),
        else => return Error.UnsupportedDType,
    }
}

pub fn symmetric(comptime dtype: types.DType, vectors: []const u8, vectors_count: usize, depth: usize, stride_bytes: usize, out: []u8, out_stride_bytes: usize, row_start: usize, row_count: usize) Error!void {
    switch (dtype) {
        .bf16 => c.nk_dots_symmetric_bf16(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .f16 => c.nk_dots_symmetric_f16(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e4m3 => c.nk_dots_symmetric_e4m3(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e5m2 => c.nk_dots_symmetric_e5m2(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e2m3 => c.nk_dots_symmetric_e2m3(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .e3m2 => c.nk_dots_symmetric_e3m2(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .f32 => c.nk_dots_symmetric_f32(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .f64 => c.nk_dots_symmetric_f64(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .i8 => c.nk_dots_symmetric_i8(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .u8 => c.nk_dots_symmetric_u8(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .i4 => c.nk_dots_symmetric_i4(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .u4 => c.nk_dots_symmetric_u4(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        .u1 => c.nk_dots_symmetric_u1(@ptrCast(@alignCast(vectors.ptr)), vectors_count, depth, stride_bytes, @ptrCast(@alignCast(out.ptr)), out_stride_bytes, row_start, row_count),
        else => return Error.UnsupportedDType,
    }
}

test "dots packed size supports all dtypes" {
    const supported = [_]types.DType{ .bf16, .f16, .e4m3, .e5m2, .e2m3, .e3m2, .f32, .f64, .i8, .u8, .i4, .u4, .u1 };
    for (supported) |dtype| {
        try std.testing.expect((try packedSize(dtype, 1, 8)) > 0);
    }
}

test "dots f32 pack packed and symmetric" {
    const a = [_]types.F32{ 1, 2 };
    const b = [_]types.F32{ 3, 4 };
    var packed_buffer: [4096]u8 = undefined;
    const size = try packedSize(.f32, 1, 2);
    try pack(.f32, std.mem.sliceAsBytes(b[0..]), 1, 2, 2 * @sizeOf(types.F32), packed_buffer[0..size]);

    var product = [_]types.F64{0};
    try computePacked(.f32, std.mem.sliceAsBytes(a[0..]), packed_buffer[0..size], 1, 1, 2, 2 * @sizeOf(types.F32), std.mem.sliceAsBytes(product[0..]), @sizeOf(types.F64));
    try std.testing.expectApproxEqAbs(@as(types.F64, 11), product[0], 1e-6);

    var gram = [_]types.F64{0};
    try symmetric(.f32, std.mem.sliceAsBytes(a[0..]), 1, 2, 2 * @sizeOf(types.F32), std.mem.sliceAsBytes(gram[0..]), @sizeOf(types.F64), 0, 1);
    try std.testing.expectApproxEqAbs(@as(types.F64, 5), gram[0], 1e-6);
}
