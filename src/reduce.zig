const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{
    EmptyInput,
    InputTooSmall,
};

pub fn Moments(comptime Sum: type, comptime SumSq: type) type {
    return struct {
        sum: Sum,
        sumsq: SumSq,
    };
}

pub fn MinMax(comptime T: type) type {
    return struct {
        min: T,
        min_index: usize,
        max: T,
        max_index: usize,
    };
}

pub fn MomentsResult(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64, .f32 => Moments(types.F64, types.F64),
        .f16, .bf16, .e4m3, .e5m2, .e2m3, .e3m2 => Moments(types.F32, types.F32),
        .i8, .i16, .i32, .i64, .i4 => Moments(types.I64, types.U64),
        .u8, .u16, .u32, .u64, .u4, .u1 => Moments(types.U64, types.U64),
        else => @compileError("unsupported dtype for reduce moments"),
    };
}

pub fn MinMaxResult(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64 => MinMax(types.F64),
        .f32 => MinMax(types.F32),
        .f16 => MinMax(types.F16),
        .bf16 => MinMax(types.BF16),
        .e4m3 => MinMax(types.E4M3),
        .e5m2 => MinMax(types.E5M2),
        .e2m3 => MinMax(types.E2M3),
        .e3m2 => MinMax(types.E3M2),
        .i8 => MinMax(types.I8),
        .u8 => MinMax(types.U8),
        .i16 => MinMax(types.I16),
        .u16 => MinMax(types.U16),
        .i32 => MinMax(types.I32),
        .u32 => MinMax(types.U32),
        .i64 => MinMax(types.I64),
        .u64 => MinMax(types.U64),
        .i4 => MinMax(types.I8),
        .u4, .u1 => MinMax(types.U8),
        else => @compileError("unsupported dtype for reduce minmax"),
    };
}

fn requiredBytes(comptime A: type, count: usize, comptime values_per_byte: usize) Error!usize {
    if (count == 0) return Error.EmptyInput;
    if (values_per_byte == 1) return std.math.mul(usize, count, @sizeOf(A)) catch Error.InputTooSmall;
    return count / values_per_byte + @intFromBool(count % values_per_byte != 0);
}

fn expectStorage(comptime A: type, data: []const u8, count: usize, comptime values_per_byte: usize) Error!void {
    const required = try requiredBytes(A, count, values_per_byte);
    if (data.len < required) return Error.InputTooSmall;
}

fn momentsImpl(comptime A: type, comptime Sum: type, comptime SumSq: type, comptime func: anytype, data: []const u8, count: usize, comptime values_per_byte: usize) Error!Moments(Sum, SumSq) {
    try expectStorage(A, data, count, values_per_byte);
    var result: Moments(Sum, SumSq) = undefined;
    func(@ptrCast(@alignCast(data.ptr)), count, @sizeOf(A), @ptrCast(&result.sum), @ptrCast(&result.sumsq));
    return result;
}

fn minmaxImpl(comptime A: type, comptime Value: type, comptime func: anytype, data: []const u8, count: usize, comptime values_per_byte: usize) Error!MinMax(Value) {
    try expectStorage(A, data, count, values_per_byte);
    var result: MinMax(Value) = undefined;
    func(@ptrCast(@alignCast(data.ptr)), count, @sizeOf(A), @ptrCast(&result.min), @ptrCast(&result.min_index), @ptrCast(&result.max), @ptrCast(&result.max_index));
    return result;
}

pub fn moments(comptime dtype: types.DType, data: []const u8, count: usize) Error!MomentsResult(dtype) {
    return switch (dtype) {
        .f64 => momentsImpl(types.F64, types.F64, types.F64, c.nk_reduce_moments_f64, data, count, 1),
        .f32 => momentsImpl(types.F32, types.F64, types.F64, c.nk_reduce_moments_f32, data, count, 1),
        .f16 => momentsImpl(types.F16, types.F32, types.F32, c.nk_reduce_moments_f16, data, count, 1),
        .bf16 => momentsImpl(types.BF16, types.F32, types.F32, c.nk_reduce_moments_bf16, data, count, 1),
        .e4m3 => momentsImpl(types.E4M3, types.F32, types.F32, c.nk_reduce_moments_e4m3, data, count, 1),
        .e5m2 => momentsImpl(types.E5M2, types.F32, types.F32, c.nk_reduce_moments_e5m2, data, count, 1),
        .e2m3 => momentsImpl(types.E2M3, types.F32, types.F32, c.nk_reduce_moments_e2m3, data, count, 1),
        .e3m2 => momentsImpl(types.E3M2, types.F32, types.F32, c.nk_reduce_moments_e3m2, data, count, 1),
        .i8 => momentsImpl(types.I8, types.I64, types.U64, c.nk_reduce_moments_i8, data, count, 1),
        .u8 => momentsImpl(types.U8, types.U64, types.U64, c.nk_reduce_moments_u8, data, count, 1),
        .i16 => momentsImpl(types.I16, types.I64, types.U64, c.nk_reduce_moments_i16, data, count, 1),
        .u16 => momentsImpl(types.U16, types.U64, types.U64, c.nk_reduce_moments_u16, data, count, 1),
        .i32 => momentsImpl(types.I32, types.I64, types.U64, c.nk_reduce_moments_i32, data, count, 1),
        .u32 => momentsImpl(types.U32, types.U64, types.U64, c.nk_reduce_moments_u32, data, count, 1),
        .i64 => momentsImpl(types.I64, types.I64, types.U64, c.nk_reduce_moments_i64, data, count, 1),
        .u64 => momentsImpl(types.U64, types.U64, types.U64, c.nk_reduce_moments_u64, data, count, 1),
        .i4 => momentsImpl(types.I4x2, types.I64, types.U64, c.nk_reduce_moments_i4, data, count, 2),
        .u4 => momentsImpl(types.U4x2, types.U64, types.U64, c.nk_reduce_moments_u4, data, count, 2),
        .u1 => momentsImpl(types.U1x8, types.U64, types.U64, c.nk_reduce_moments_u1, data, count, 8),
        else => @compileError("unsupported dtype for reduce moments"),
    };
}

pub fn minmax(comptime dtype: types.DType, data: []const u8, count: usize) Error!MinMaxResult(dtype) {
    return switch (dtype) {
        .f64 => minmaxImpl(types.F64, types.F64, c.nk_reduce_minmax_f64, data, count, 1),
        .f32 => minmaxImpl(types.F32, types.F32, c.nk_reduce_minmax_f32, data, count, 1),
        .f16 => minmaxImpl(types.F16, types.F16, c.nk_reduce_minmax_f16, data, count, 1),
        .bf16 => minmaxImpl(types.BF16, types.BF16, c.nk_reduce_minmax_bf16, data, count, 1),
        .e4m3 => minmaxImpl(types.E4M3, types.E4M3, c.nk_reduce_minmax_e4m3, data, count, 1),
        .e5m2 => minmaxImpl(types.E5M2, types.E5M2, c.nk_reduce_minmax_e5m2, data, count, 1),
        .e2m3 => minmaxImpl(types.E2M3, types.E2M3, c.nk_reduce_minmax_e2m3, data, count, 1),
        .e3m2 => minmaxImpl(types.E3M2, types.E3M2, c.nk_reduce_minmax_e3m2, data, count, 1),
        .i8 => minmaxImpl(types.I8, types.I8, c.nk_reduce_minmax_i8, data, count, 1),
        .u8 => minmaxImpl(types.U8, types.U8, c.nk_reduce_minmax_u8, data, count, 1),
        .i16 => minmaxImpl(types.I16, types.I16, c.nk_reduce_minmax_i16, data, count, 1),
        .u16 => minmaxImpl(types.U16, types.U16, c.nk_reduce_minmax_u16, data, count, 1),
        .i32 => minmaxImpl(types.I32, types.I32, c.nk_reduce_minmax_i32, data, count, 1),
        .u32 => minmaxImpl(types.U32, types.U32, c.nk_reduce_minmax_u32, data, count, 1),
        .i64 => minmaxImpl(types.I64, types.I64, c.nk_reduce_minmax_i64, data, count, 1),
        .u64 => minmaxImpl(types.U64, types.U64, c.nk_reduce_minmax_u64, data, count, 1),
        .i4 => minmaxImpl(types.I4x2, types.I8, c.nk_reduce_minmax_i4, data, count, 2),
        .u4 => minmaxImpl(types.U4x2, types.U8, c.nk_reduce_minmax_u4, data, count, 2),
        .u1 => minmaxImpl(types.U1x8, types.U8, c.nk_reduce_minmax_u1, data, count, 8),
        else => @compileError("unsupported dtype for reduce minmax"),
    };
}

fn expectZeroMoment(comptime dtype: types.DType, data: []const u8, count: usize) !void {
    const result = try moments(dtype, data, count);
    switch (@TypeOf(result.sum)) {
        types.F64 => {
            try std.testing.expectApproxEqAbs(@as(types.F64, 0), result.sum, 1e-12);
            try std.testing.expectApproxEqAbs(@as(types.F64, 0), result.sumsq, 1e-12);
        },
        types.F32 => {
            try std.testing.expectApproxEqAbs(@as(types.F32, 0), result.sum, 1e-6);
            try std.testing.expectApproxEqAbs(@as(types.F32, 0), result.sumsq, 1e-6);
        },
        types.I64 => {
            try std.testing.expectEqual(@as(types.I64, 0), result.sum);
            try std.testing.expectEqual(@as(types.U64, 0), result.sumsq);
        },
        types.U64 => {
            try std.testing.expectEqual(@as(types.U64, 0), result.sum);
            try std.testing.expectEqual(@as(types.U64, 0), result.sumsq);
        },
        else => unreachable,
    }
}

fn expectSingleElementMinMax(comptime dtype: types.DType, data: []const u8, count: usize) !void {
    const result = try minmax(dtype, data, count);
    try std.testing.expectEqual(@as(usize, 0), result.min_index);
    try std.testing.expectEqual(@as(usize, 0), result.max_index);
}

test "reduce comptime dtype API returns exact floating results" {
    const xs_f32 = [_]types.F32{ 1, 2, 3 };
    const m_f32 = try moments(.f32, std.mem.sliceAsBytes(xs_f32[0..]), xs_f32.len);
    try std.testing.expectEqual(Moments(types.F64, types.F64), @TypeOf(m_f32));
    try std.testing.expectApproxEqAbs(@as(types.F64, 6), m_f32.sum, 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 14), m_f32.sumsq, 1e-6);
    const mm_f32 = try minmax(.f32, std.mem.sliceAsBytes(xs_f32[0..]), xs_f32.len);
    try std.testing.expectEqual(MinMax(types.F32), @TypeOf(mm_f32));
    try std.testing.expectEqual(@as(types.F32, 1), mm_f32.min);
    try std.testing.expectEqual(@as(usize, 0), mm_f32.min_index);
    try std.testing.expectEqual(@as(types.F32, 3), mm_f32.max);
    try std.testing.expectEqual(@as(usize, 2), mm_f32.max_index);

    const xs_f64 = [_]types.F64{ 1, 2, 3 };
    try std.testing.expectApproxEqAbs(@as(types.F64, 6), (try moments(.f64, std.mem.sliceAsBytes(xs_f64[0..]), xs_f64.len)).sum, 1e-12);
    try std.testing.expectEqual(@as(types.F64, 3), (try minmax(.f64, std.mem.sliceAsBytes(xs_f64[0..]), xs_f64.len)).max);
}

test "reduce comptime dtype API covers all zero dtypes" {
    const cast = @import("cast.zig");
    const zero_f16 = [_]types.F16{cast.fromF32(.f16, 0)};
    const zero_bf16 = [_]types.BF16{cast.fromF32(.bf16, 0)};
    const zero_e4m3 = [_]types.E4M3{cast.fromF32(.e4m3, 0)};
    const zero_e5m2 = [_]types.E5M2{cast.fromF32(.e5m2, 0)};
    const zero_e2m3 = [_]types.E2M3{cast.fromF32(.e2m3, 0)};
    const zero_e3m2 = [_]types.E3M2{cast.fromF32(.e3m2, 0)};
    const zero_i8 = [_]types.I8{0};
    const zero_u8 = [_]types.U8{0};
    const zero_i16 = [_]types.I16{0};
    const zero_u16 = [_]types.U16{0};
    const zero_i32 = [_]types.I32{0};
    const zero_u32 = [_]types.U32{0};
    const zero_i64 = [_]types.I64{0};
    const zero_u64 = [_]types.U64{0};
    const zero_i4 = [_]types.I4x2{0};
    const zero_u4 = [_]types.U4x2{0};
    const zero_u1 = [_]types.U1x8{0};

    try expectZeroMoment(.f16, std.mem.sliceAsBytes(zero_f16[0..]), 1);
    try expectZeroMoment(.bf16, std.mem.sliceAsBytes(zero_bf16[0..]), 1);
    try expectZeroMoment(.e4m3, std.mem.sliceAsBytes(zero_e4m3[0..]), 1);
    try expectZeroMoment(.e5m2, std.mem.sliceAsBytes(zero_e5m2[0..]), 1);
    try expectZeroMoment(.e2m3, std.mem.sliceAsBytes(zero_e2m3[0..]), 1);
    try expectZeroMoment(.e3m2, std.mem.sliceAsBytes(zero_e3m2[0..]), 1);
    try expectZeroMoment(.i8, std.mem.sliceAsBytes(zero_i8[0..]), 1);
    try expectZeroMoment(.u8, std.mem.sliceAsBytes(zero_u8[0..]), 1);
    try expectZeroMoment(.i16, std.mem.sliceAsBytes(zero_i16[0..]), 1);
    try expectZeroMoment(.u16, std.mem.sliceAsBytes(zero_u16[0..]), 1);
    try expectZeroMoment(.i32, std.mem.sliceAsBytes(zero_i32[0..]), 1);
    try expectZeroMoment(.u32, std.mem.sliceAsBytes(zero_u32[0..]), 1);
    try expectZeroMoment(.i64, std.mem.sliceAsBytes(zero_i64[0..]), 1);
    try expectZeroMoment(.u64, std.mem.sliceAsBytes(zero_u64[0..]), 1);
    try expectZeroMoment(.i4, std.mem.sliceAsBytes(zero_i4[0..]), 2);
    try expectZeroMoment(.u4, std.mem.sliceAsBytes(zero_u4[0..]), 2);
    try expectZeroMoment(.u1, std.mem.sliceAsBytes(zero_u1[0..]), 8);

    try expectSingleElementMinMax(.f16, std.mem.sliceAsBytes(zero_f16[0..]), 1);
    try expectSingleElementMinMax(.bf16, std.mem.sliceAsBytes(zero_bf16[0..]), 1);
    try expectSingleElementMinMax(.e4m3, std.mem.sliceAsBytes(zero_e4m3[0..]), 1);
    try expectSingleElementMinMax(.e5m2, std.mem.sliceAsBytes(zero_e5m2[0..]), 1);
    try expectSingleElementMinMax(.e2m3, std.mem.sliceAsBytes(zero_e2m3[0..]), 1);
    try expectSingleElementMinMax(.e3m2, std.mem.sliceAsBytes(zero_e3m2[0..]), 1);
    try expectSingleElementMinMax(.i8, std.mem.sliceAsBytes(zero_i8[0..]), 1);
    try expectSingleElementMinMax(.u8, std.mem.sliceAsBytes(zero_u8[0..]), 1);
    try expectSingleElementMinMax(.i16, std.mem.sliceAsBytes(zero_i16[0..]), 1);
    try expectSingleElementMinMax(.u16, std.mem.sliceAsBytes(zero_u16[0..]), 1);
    try expectSingleElementMinMax(.i32, std.mem.sliceAsBytes(zero_i32[0..]), 1);
    try expectSingleElementMinMax(.u32, std.mem.sliceAsBytes(zero_u32[0..]), 1);
    try expectSingleElementMinMax(.i64, std.mem.sliceAsBytes(zero_i64[0..]), 1);
    try expectSingleElementMinMax(.u64, std.mem.sliceAsBytes(zero_u64[0..]), 1);
    try expectSingleElementMinMax(.i4, std.mem.sliceAsBytes(zero_i4[0..]), 2);
    try expectSingleElementMinMax(.u4, std.mem.sliceAsBytes(zero_u4[0..]), 2);
    try expectSingleElementMinMax(.u1, std.mem.sliceAsBytes(zero_u1[0..]), 8);
}

test "reduce comptime dtype API reports invalid inputs" {
    const zero = [_]types.F32{0};
    try std.testing.expectError(Error.EmptyInput, moments(.f32, std.mem.sliceAsBytes(zero[0..]), 0));
    try std.testing.expectError(Error.InputTooSmall, minmax(.f32, std.mem.sliceAsBytes(zero[0..])[0..3], 1));
}
