const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{
    EmptyInput,
    InputTooSmall,
};

pub fn BilinearResult(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64, .f32 => types.F64,
        .f16, .bf16 => types.F32,
        .f64c, .f32c => types.F64C,
        .f16c, .bf16c => types.F32C,
        else => @compileError("unsupported dtype for bilinear"),
    };
}

pub fn MahalanobisResult(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64, .f32 => types.F64,
        .f16, .bf16 => types.F32,
        else => @compileError("unsupported dtype for mahalanobis"),
    };
}

fn vectorBytes(comptime A: type, count: usize) Error!usize {
    if (count == 0) return Error.EmptyInput;
    return std.math.mul(usize, count, @sizeOf(A)) catch Error.InputTooSmall;
}

fn matrixBytes(comptime A: type, count: usize) Error!usize {
    const cells = std.math.mul(usize, count, count) catch return Error.InputTooSmall;
    return std.math.mul(usize, cells, @sizeOf(A)) catch Error.InputTooSmall;
}

fn formImpl(comptime A: type, comptime R: type, comptime func: anytype, a: []const u8, b: []const u8, matrix: []const u8, count: usize) Error!R {
    const vector_required = try vectorBytes(A, count);
    const matrix_required = try matrixBytes(A, count);
    if (a.len < vector_required or b.len < vector_required or matrix.len < matrix_required) return Error.InputTooSmall;
    var result: R = undefined;
    func(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), @ptrCast(@alignCast(matrix.ptr)), count, @ptrCast(&result));
    return result;
}

pub fn bilinear(comptime dtype: types.DType, a: []const u8, b: []const u8, matrix: []const u8, count: usize) Error!BilinearResult(dtype) {
    return switch (dtype) {
        .f64 => formImpl(types.F64, types.F64, c.nk_bilinear_f64, a, b, matrix, count),
        .f32 => formImpl(types.F32, types.F64, c.nk_bilinear_f32, a, b, matrix, count),
        .f16 => formImpl(types.F16, types.F32, c.nk_bilinear_f16, a, b, matrix, count),
        .bf16 => formImpl(types.BF16, types.F32, c.nk_bilinear_bf16, a, b, matrix, count),
        .f64c => formImpl(types.F64C, types.F64C, c.nk_bilinear_f64c, a, b, matrix, count),
        .f32c => formImpl(types.F32C, types.F64C, c.nk_bilinear_f32c, a, b, matrix, count),
        .f16c => formImpl(types.F16C, types.F32C, c.nk_bilinear_f16c, a, b, matrix, count),
        .bf16c => formImpl(types.BF16C, types.F32C, c.nk_bilinear_bf16c, a, b, matrix, count),
        else => @compileError("unsupported dtype for bilinear"),
    };
}

pub fn mahalanobis(comptime dtype: types.DType, a: []const u8, b: []const u8, matrix: []const u8, count: usize) Error!MahalanobisResult(dtype) {
    return switch (dtype) {
        .f64 => formImpl(types.F64, types.F64, c.nk_mahalanobis_f64, a, b, matrix, count),
        .f32 => formImpl(types.F32, types.F64, c.nk_mahalanobis_f32, a, b, matrix, count),
        .f16 => formImpl(types.F16, types.F32, c.nk_mahalanobis_f16, a, b, matrix, count),
        .bf16 => formImpl(types.BF16, types.F32, c.nk_mahalanobis_bf16, a, b, matrix, count),
        else => @compileError("unsupported dtype for mahalanobis"),
    };
}

test "curved comptime dtype API returns exact real results" {
    const xs_f32 = [_]types.F32{ 1, 2 };
    const id_f32 = [_]types.F32{ 1, 0, 0, 1 };
    const xs_bytes = std.mem.sliceAsBytes(xs_f32[0..]);
    const id_bytes = std.mem.sliceAsBytes(id_f32[0..]);
    const bilinear_result = try bilinear(.f32, xs_bytes, xs_bytes, id_bytes, xs_f32.len);
    try std.testing.expectEqual(types.F64, @TypeOf(bilinear_result));
    try std.testing.expectApproxEqAbs(@as(types.F64, 5), bilinear_result, 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), try mahalanobis(.f32, xs_bytes, xs_bytes, id_bytes, xs_f32.len), 1e-6);
}

test "curved comptime dtype API covers real dtypes" {
    const cast = @import("cast.zig");
    const xs_f64 = [_]types.F64{ 1, 2 };
    const id_f64 = [_]types.F64{ 1, 0, 0, 1 };
    try std.testing.expectApproxEqAbs(@as(types.F64, 5), try bilinear(.f64, std.mem.sliceAsBytes(xs_f64[0..]), std.mem.sliceAsBytes(xs_f64[0..]), std.mem.sliceAsBytes(id_f64[0..]), xs_f64.len), 1e-12);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), try mahalanobis(.f64, std.mem.sliceAsBytes(xs_f64[0..]), std.mem.sliceAsBytes(xs_f64[0..]), std.mem.sliceAsBytes(id_f64[0..]), xs_f64.len), 1e-12);

    const xs_f16 = [_]types.F16{ cast.fromF32(.f16, 1), cast.fromF32(.f16, 2) };
    const id_f16 = [_]types.F16{ cast.fromF32(.f16, 1), cast.fromF32(.f16, 0), cast.fromF32(.f16, 0), cast.fromF32(.f16, 1) };
    try std.testing.expectApproxEqAbs(@as(types.F32, 5), try bilinear(.f16, std.mem.sliceAsBytes(xs_f16[0..]), std.mem.sliceAsBytes(xs_f16[0..]), std.mem.sliceAsBytes(id_f16[0..]), xs_f16.len), 1e-3);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), try mahalanobis(.f16, std.mem.sliceAsBytes(xs_f16[0..]), std.mem.sliceAsBytes(xs_f16[0..]), std.mem.sliceAsBytes(id_f16[0..]), xs_f16.len), 1e-3);

    const xs_bf16 = [_]types.BF16{ cast.fromF32(.bf16, 1), cast.fromF32(.bf16, 2) };
    const id_bf16 = [_]types.BF16{ cast.fromF32(.bf16, 1), cast.fromF32(.bf16, 0), cast.fromF32(.bf16, 0), cast.fromF32(.bf16, 1) };
    try std.testing.expectApproxEqAbs(@as(types.F32, 5), try bilinear(.bf16, std.mem.sliceAsBytes(xs_bf16[0..]), std.mem.sliceAsBytes(xs_bf16[0..]), std.mem.sliceAsBytes(id_bf16[0..]), xs_bf16.len), 1e-3);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), try mahalanobis(.bf16, std.mem.sliceAsBytes(xs_bf16[0..]), std.mem.sliceAsBytes(xs_bf16[0..]), std.mem.sliceAsBytes(id_bf16[0..]), xs_bf16.len), 1e-3);
}

test "curved comptime dtype API covers complex bilinear dtypes" {
    const cast = @import("cast.zig");
    const zero_f64c = [_]types.F64C{.{ .real = 0, .imag = 0 }};
    const zero_f32c = [_]types.F32C{.{ .real = 0, .imag = 0 }};
    const zero_f16c = [_]types.F16C{.{ .real = cast.fromF32(.f16, 0), .imag = cast.fromF32(.f16, 0) }};
    const zero_bf16c = [_]types.BF16C{.{ .real = cast.fromF32(.bf16, 0), .imag = cast.fromF32(.bf16, 0) }};

    try std.testing.expectEqual(@as(types.F64, 0), (try bilinear(.f64c, std.mem.sliceAsBytes(zero_f64c[0..]), std.mem.sliceAsBytes(zero_f64c[0..]), std.mem.sliceAsBytes(zero_f64c[0..]), 1)).real);
    try std.testing.expectEqual(@as(types.F64, 0), (try bilinear(.f32c, std.mem.sliceAsBytes(zero_f32c[0..]), std.mem.sliceAsBytes(zero_f32c[0..]), std.mem.sliceAsBytes(zero_f32c[0..]), 1)).real);
    try std.testing.expectEqual(@as(types.F32, 0), (try bilinear(.f16c, std.mem.sliceAsBytes(zero_f16c[0..]), std.mem.sliceAsBytes(zero_f16c[0..]), std.mem.sliceAsBytes(zero_f16c[0..]), 1)).real);
    try std.testing.expectEqual(@as(types.F32, 0), (try bilinear(.bf16c, std.mem.sliceAsBytes(zero_bf16c[0..]), std.mem.sliceAsBytes(zero_bf16c[0..]), std.mem.sliceAsBytes(zero_bf16c[0..]), 1)).real);
}

test "curved comptime dtype API reports invalid inputs" {
    const xs = [_]types.F32{1};
    const id = [_]types.F32{1};
    try std.testing.expectError(Error.EmptyInput, bilinear(.f32, std.mem.sliceAsBytes(xs[0..]), std.mem.sliceAsBytes(xs[0..]), std.mem.sliceAsBytes(id[0..]), 0));
    try std.testing.expectError(Error.InputTooSmall, mahalanobis(.f32, std.mem.sliceAsBytes(xs[0..])[0..3], std.mem.sliceAsBytes(xs[0..]), std.mem.sliceAsBytes(id[0..]), 1));
}
