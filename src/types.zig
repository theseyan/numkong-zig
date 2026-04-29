const c = @import("c.zig").raw;

pub const U1x8 = u8;
pub const I4x2 = u8;
pub const U4x2 = u8;
pub const E4M3 = u8;
pub const E5M2 = u8;
pub const E2M3 = u8;
pub const E3M2 = u8;
pub const I8 = i8;
pub const U8 = u8;
pub const I16 = i16;
pub const U16 = u16;
pub const I32 = i32;
pub const U32 = u32;
pub const I64 = i64;
pub const U64 = u64;
pub const F16 = u16;
pub const BF16 = u16;
pub const F32 = f32;
pub const F64 = f64;
pub const Size = usize;
pub const SSize = isize;
pub const Capability = u64;
pub const KernelFunction = *const fn (?*anyopaque) callconv(.c) void;

pub const KernelLookup = struct {
    function: ?KernelFunction,
    capability: Capability,
};

pub const F16C = extern struct { real: F16, imag: F16 };
pub const BF16C = extern struct { real: BF16, imag: BF16 };
pub const F32C = extern struct { real: F32, imag: F32 };
pub const F64C = extern struct { real: F64, imag: F64 };

pub const Version = struct {
    pub const major = c.NK_VERSION_MAJOR;
    pub const minor = c.NK_VERSION_MINOR;
    pub const patch = c.NK_VERSION_PATCH;
};

pub const DType = enum(u32) {
    unknown = c.nk_dtype_unknown_k,
    u1 = c.nk_u1_k,
    i8 = c.nk_i8_k,
    i16 = c.nk_i16_k,
    i32 = c.nk_i32_k,
    i64 = c.nk_i64_k,
    u8 = c.nk_u8_k,
    u16 = c.nk_u16_k,
    u32 = c.nk_u32_k,
    u64 = c.nk_u64_k,
    f64 = c.nk_f64_k,
    f32 = c.nk_f32_k,
    f16 = c.nk_f16_k,
    bf16 = c.nk_bf16_k,
    e4m3 = c.nk_e4m3_k,
    e5m2 = c.nk_e5m2_k,
    i4 = c.nk_i4_k,
    u4 = c.nk_u4_k,
    e2m3 = c.nk_e2m3_k,
    e3m2 = c.nk_e3m2_k,
    f64c = c.nk_f64c_k,
    f32c = c.nk_f32c_k,
    f16c = c.nk_f16c_k,
    bf16c = c.nk_bf16c_k,
};

pub const DTypeFamily = enum(u32) {
    unknown = c.nk_dtype_family_unknown_k,
    float = c.nk_dtype_family_float_k,
    complex_float = c.nk_dtype_family_complex_float_k,
    int = c.nk_dtype_family_int_k,
    uint = c.nk_dtype_family_uint_k,
};

pub const KernelKind = enum(u32) {
    unknown = c.nk_kernel_unknown_k,
    dot = c.nk_kernel_dot_k,
    vdot = c.nk_kernel_vdot_k,
    angular = c.nk_kernel_angular_k,
    euclidean = c.nk_kernel_euclidean_k,
    sqeuclidean = c.nk_kernel_sqeuclidean_k,
    hamming = c.nk_kernel_hamming_k,
    jaccard = c.nk_kernel_jaccard_k,
    bilinear = c.nk_kernel_bilinear_k,
    mahalanobis = c.nk_kernel_mahalanobis_k,
    haversine = c.nk_kernel_haversine_k,
    vincenty = c.nk_kernel_vincenty_k,
    kld = c.nk_kernel_kld_k,
    jsd = c.nk_kernel_jsd_k,
    rmsd = c.nk_kernel_rmsd_k,
    kabsch = c.nk_kernel_kabsch_k,
    umeyama = c.nk_kernel_umeyama_k,
    sparse_dot = c.nk_kernel_sparse_dot_k,
    sparse_intersect = c.nk_kernel_sparse_intersect_k,
    each_scale = c.nk_kernel_each_scale_k,
    each_sum = c.nk_kernel_each_sum_k,
    each_blend = c.nk_kernel_each_blend_k,
    each_fma = c.nk_kernel_each_fma_k,
    each_sin = c.nk_kernel_each_sin_k,
    each_cos = c.nk_kernel_each_cos_k,
    each_atan = c.nk_kernel_each_atan_k,
    reduce_moments = c.nk_kernel_reduce_moments_k,
    reduce_minmax = c.nk_kernel_reduce_minmax_k,
    dots_packed_size = c.nk_kernel_dots_packed_size_k,
    dots_pack = c.nk_kernel_dots_pack_k,
    dots_packed = c.nk_kernel_dots_packed_k,
    dots_symmetric = c.nk_kernel_dots_symmetric_k,
    hammings_packed = c.nk_kernel_hammings_packed_k,
    hammings_symmetric = c.nk_kernel_hammings_symmetric_k,
    jaccards_packed = c.nk_kernel_jaccards_packed_k,
    jaccards_symmetric = c.nk_kernel_jaccards_symmetric_k,
    angulars_packed = c.nk_kernel_angulars_packed_k,
    angulars_symmetric = c.nk_kernel_angulars_symmetric_k,
    euclideans_packed = c.nk_kernel_euclideans_packed_k,
    euclideans_symmetric = c.nk_kernel_euclideans_symmetric_k,
    maxsim_packed_size = c.nk_kernel_maxsim_packed_size_k,
    maxsim_pack = c.nk_kernel_maxsim_pack_k,
    maxsim_packed = c.nk_kernel_maxsim_packed_k,
    cast = c.nk_kernel_cast_k,
};

pub fn capabilities() Capability {
    return @intCast(c.nk_capabilities());
}

pub fn capabilitiesAvailable() Capability {
    return @intCast(c.nk_capabilities_available());
}

pub fn capabilitiesCompiled() Capability {
    return @intCast(c.nk_capabilities_compiled());
}

pub fn usesDynamicDispatch() bool {
    return c.nk_uses_dynamic_dispatch() != 0;
}

pub fn configureThread(capability: Capability) !void {
    if (c.nk_configure_thread(capability) != 0) return error.UnsupportedCapability;
}

pub fn updateDispatchTable(capabilities_mask: Capability) void {
    c.nk_dispatch_table_update(capabilities_mask);
}

pub fn findKernel(kind: KernelKind, dtype: DType, viable: Capability) KernelLookup {
    var function: c.nk_kernel_punned_t = null;
    var capability: c.nk_capability_t = 0;
    c.nk_find_kernel_punned(@intFromEnum(kind), @intFromEnum(dtype), viable, &function, &capability);
    return .{
        .function = function,
        .capability = @intCast(capability),
    };
}

pub fn dtypeFamily(dtype: DType) DTypeFamily {
    return @enumFromInt(c.nk_dtype_family(@intFromEnum(dtype)));
}

pub fn dtypeBits(dtype: DType) Size {
    return @intCast(c.nk_dtype_bits(@intFromEnum(dtype)));
}

pub fn dimensionsPerValue(dtype: DType) Size {
    return @intCast(c.nk_dimensions_per_value(@intFromEnum(dtype)));
}

pub fn kernelOutputDType(kind: KernelKind, input: DType) DType {
    return @enumFromInt(c.nk_kernel_output_dtype(@intFromEnum(kind), @intFromEnum(input)));
}

test "metadata" {
    const std = @import("std");
    try std.testing.expectEqual(@as(usize, 32), dtypeBits(.f32));
    try std.testing.expectEqual(DTypeFamily.float, dtypeFamily(.f32));
    try std.testing.expectEqual(DType.f64, kernelOutputDType(.dot, .f32));
    try std.testing.expect(usesDynamicDispatch());
}

test "dispatch table lookup" {
    const std = @import("std");
    const caps = capabilitiesAvailable();
    updateDispatchTable(caps);
    const lookup = findKernel(.dot, .f32, caps);
    try std.testing.expect(lookup.function != null);
    try std.testing.expect(lookup.capability != 0);
}
