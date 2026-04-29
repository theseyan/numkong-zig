const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub fn Index(comptime dtype: types.DType) type {
    return switch (dtype) {
        .u16 => types.U16,
        .u32 => types.U32,
        .u64 => types.U64,
        else => @compileError("unsupported sparse index dtype"),
    };
}

pub fn Weight(comptime dtype: types.DType) type {
    return switch (dtype) {
        .bf16 => types.BF16,
        .f32 => types.F32,
        else => @compileError("unsupported sparse weight dtype"),
    };
}

pub fn DotResult(comptime index_dtype: types.DType, comptime weight_dtype: types.DType) type {
    return switch (index_dtype) {
        .u16 => switch (weight_dtype) {
            .bf16 => types.F32,
            else => @compileError("unsupported sparse dot dtype combination"),
        },
        .u32 => switch (weight_dtype) {
            .f32 => types.F64,
            else => @compileError("unsupported sparse dot dtype combination"),
        },
        else => @compileError("unsupported sparse dot dtype combination"),
    };
}

fn intersectImpl(comptime T: type, comptime func: anytype, a: []const T, b: []const T, out: ?[]T) usize {
    if (out) |buffer| std.debug.assert(buffer.len >= @min(a.len, b.len));
    var count: usize = undefined;
    const out_ptr = if (out) |buffer| buffer.ptr else null;
    func(@ptrCast(a.ptr), @ptrCast(b.ptr), a.len, b.len, @ptrCast(out_ptr), @ptrCast(&count));
    return count;
}

pub fn intersect(comptime index_dtype: types.DType, a: []const Index(index_dtype), b: []const Index(index_dtype), out: ?[]Index(index_dtype)) usize {
    return switch (index_dtype) {
        .u16 => intersectImpl(types.U16, c.nk_sparse_intersect_u16, a, b, out),
        .u32 => intersectImpl(types.U32, c.nk_sparse_intersect_u32, a, b, out),
        .u64 => intersectImpl(types.U64, c.nk_sparse_intersect_u64, a, b, out),
        else => unreachable,
    };
}

pub fn dot(
    comptime index_dtype: types.DType,
    comptime weight_dtype: types.DType,
    a: []const Index(index_dtype),
    b: []const Index(index_dtype),
    a_weights: []const Weight(weight_dtype),
    b_weights: []const Weight(weight_dtype),
) DotResult(index_dtype, weight_dtype) {
    std.debug.assert(a.len == a_weights.len);
    std.debug.assert(b.len == b_weights.len);
    return switch (index_dtype) {
        .u16 => switch (weight_dtype) {
            .bf16 => blk: {
                var product: types.F32 = undefined;
                c.nk_sparse_dot_u16bf16(@ptrCast(a.ptr), @ptrCast(b.ptr), @ptrCast(a_weights.ptr), @ptrCast(b_weights.ptr), a.len, b.len, @ptrCast(&product));
                break :blk product;
            },
            else => unreachable,
        },
        .u32 => switch (weight_dtype) {
            .f32 => blk: {
                var product: types.F64 = undefined;
                c.nk_sparse_dot_u32f32(@ptrCast(a.ptr), @ptrCast(b.ptr), a_weights.ptr, b_weights.ptr, a.len, b.len, @ptrCast(&product));
                break :blk product;
            },
            else => unreachable,
        },
        else => unreachable,
    };
}

test "sparse intersections count and write output" {
    var out16 = [_]types.U16{ 0, 0, 0 };
    const count16 = intersect(.u16, &.{ 1, 3, 5 }, &.{ 3, 4, 5 }, &out16);
    try std.testing.expectEqual(@as(usize, 2), count16);
    try std.testing.expectEqualSlices(types.U16, &.{ 3, 5 }, out16[0..count16]);
    try std.testing.expectEqual(@as(usize, 2), intersect(.u16, &.{ 1, 3, 5 }, &.{ 3, 4, 5 }, null));

    var out32 = [_]types.U32{ 0, 0, 0 };
    const count32 = intersect(.u32, &.{ 1, 3, 5 }, &.{ 3, 4, 5 }, &out32);
    try std.testing.expectEqual(@as(usize, 2), count32);
    try std.testing.expectEqualSlices(types.U32, &.{ 3, 5 }, out32[0..count32]);

    var out64 = [_]types.U64{ 0, 0, 0 };
    const count64 = intersect(.u64, &.{ 1, 3, 5 }, &.{ 3, 4, 5 }, &out64);
    try std.testing.expectEqual(@as(usize, 2), count64);
    try std.testing.expectEqualSlices(types.U64, &.{ 3, 5 }, out64[0..count64]);
}

test "sparse dot products" {
    const cast = @import("cast.zig");
    const indices16_a = [_]types.U16{ 1, 3, 5 };
    const indices16_b = [_]types.U16{ 3, 5, 7 };
    const weights16_a = [_]types.BF16{ cast.fromF32(.bf16, 2), cast.fromF32(.bf16, 3), cast.fromF32(.bf16, 4) };
    const weights16_b = [_]types.BF16{ cast.fromF32(.bf16, 5), cast.fromF32(.bf16, 6), cast.fromF32(.bf16, 7) };
    try std.testing.expectApproxEqAbs(@as(types.F32, 39), dot(.u16, .bf16, &indices16_a, &indices16_b, &weights16_a, &weights16_b), 1e-3);

    const indices32_a = [_]types.U32{ 1, 3, 5 };
    const indices32_b = [_]types.U32{ 3, 5, 7 };
    const weights32_a = [_]types.F32{ 2, 3, 4 };
    const weights32_b = [_]types.F32{ 5, 6, 7 };
    const product = dot(.u32, .f32, &indices32_a, &indices32_b, &weights32_a, &weights32_b);
    try std.testing.expectEqual(types.F64, @TypeOf(product));
    try std.testing.expectApproxEqAbs(@as(types.F64, 39), product, 1e-6);
}
