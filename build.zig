const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const upstream = b.dependency("numkong", .{});

    const translate_c = b.addTranslateC(.{
        .root_source_file = upstream.path("include/numkong/numkong.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    translate_c.addIncludePath(upstream.path("include"));
    translate_c.defineCMacro("NK_DYNAMIC_DISPATCH", "1");
    translate_c.defineCMacro("NK_NATIVE_F16", "0");
    translate_c.defineCMacro("NK_NATIVE_BF16", "0");
    // Keep translate-c focused on dynamic-dispatch declarations. If target
    // kernels are enabled here, Clang exposes large intrinsic bodies that Zig's
    // translator can't represent reliably.
    addTargetFeatureMacros(translate_c, "0");

    const c_mod = translate_c.createModule();

    const c_library_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    c_library_mod.addIncludePath(upstream.path("include"));
    c_library_mod.addCSourceFiles(.{
        .root = upstream.path(""),
        .files = &numkong_c_sources,
        .flags = numkongCFlags(target),
        .language = .c,
    });

    const c_library = b.addLibrary(.{
        .name = "numkong-c",
        .root_module = c_library_mod,
    });

    const mod = b.addModule("numkong", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "numkong_c", .module = c_mod },
        },
    });
    mod.linkLibrary(c_library);

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);

    const zbench_mod = b.createModule(.{
        .root_source_file = b.dependency("zbench", .{}).path("src/zbench.zig"),
        .target = target,
        .optimize = optimize,
    });
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "numkong", .module = mod },
            .{ .name = "zbench", .module = zbench_mod },
        },
    });
    const bench_exe = b.addExecutable(.{
        .name = "numkong-bench",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run NumKong binding benchmarks");
    bench_step.dependOn(&run_bench.step);
}

const numkong_c_sources = [_][]const u8{
    "c/numkong.c",
    "c/dispatch_f64c.c",
    "c/dispatch_f32c.c",
    "c/dispatch_bf16c.c",
    "c/dispatch_f16c.c",
    "c/dispatch_f64.c",
    "c/dispatch_f32.c",
    "c/dispatch_bf16.c",
    "c/dispatch_f16.c",
    "c/dispatch_e5m2.c",
    "c/dispatch_e4m3.c",
    "c/dispatch_e3m2.c",
    "c/dispatch_e2m3.c",
    "c/dispatch_i64.c",
    "c/dispatch_i32.c",
    "c/dispatch_i16.c",
    "c/dispatch_i8.c",
    "c/dispatch_i4.c",
    "c/dispatch_u64.c",
    "c/dispatch_u32.c",
    "c/dispatch_u16.c",
    "c/dispatch_u8.c",
    "c/dispatch_u4.c",
    "c/dispatch_u1.c",
    "c/dispatch_other.c",
};

fn numkongCFlags(target: std.Build.ResolvedTarget) []const []const u8 {
    if (target.result.cpu.arch == .aarch64 and target.result.os.tag == .macos) {
        return &numkong_c_flags_macos_aarch64;
    }
    return &numkong_c_flags_base;
}

const numkong_c_flags_base = [_][]const u8{
    "-std=c99",
    "-DNK_DYNAMIC_DISPATCH=1",
    "-DNK_NATIVE_F16=0",
    "-DNK_NATIVE_BF16=0",
};

// Apple's arm64 ABI/toolchain currently fails to link NumKong's SME-family
// backends due to unresolved SME runtime state symbols
const numkong_c_flags_macos_aarch64 = numkong_c_flags_base ++ [_][]const u8{
    "-DNK_TARGET_SME=0",
    "-DNK_TARGET_SME2=0",
    "-DNK_TARGET_SME2P1=0",
    "-DNK_TARGET_SMEF64=0",
    "-DNK_TARGET_SMEHALF=0",
    "-DNK_TARGET_SMEBF16=0",
    "-DNK_TARGET_SMEBI32=0",
    "-DNK_TARGET_SMELUT2=0",
    "-DNK_TARGET_SMEFA64=0",
};

fn addTargetFeatureMacros(translate_c: *std.Build.Step.TranslateC, value: []const u8) void {
    for (target_feature_macros) |feature| {
        translate_c.defineCMacro(feature, value);
    }
}

const target_feature_macros = [_][]const u8{
    "NK_TARGET_HASWELL",
    "NK_TARGET_SKYLAKE",
    "NK_TARGET_ICELAKE",
    "NK_TARGET_GENOA",
    "NK_TARGET_SAPPHIRE",
    "NK_TARGET_SAPPHIREAMX",
    "NK_TARGET_GRANITEAMX",
    "NK_TARGET_DIAMOND",
    "NK_TARGET_TURIN",
    "NK_TARGET_ALDER",
    "NK_TARGET_SIERRA",
    "NK_TARGET_NEON",
    "NK_TARGET_NEONHALF",
    "NK_TARGET_NEONSDOT",
    "NK_TARGET_NEONBFDOT",
    "NK_TARGET_NEONFHM",
    "NK_TARGET_NEONFP8",
    "NK_TARGET_SVE",
    "NK_TARGET_SVEHALF",
    "NK_TARGET_SVEBFDOT",
    "NK_TARGET_SVESDOT",
    "NK_TARGET_SVE2",
    "NK_TARGET_SVE2P1",
    "NK_TARGET_SME",
    "NK_TARGET_SME2",
    "NK_TARGET_SME2P1",
    "NK_TARGET_SMEF64",
    "NK_TARGET_SMEHALF",
    "NK_TARGET_SMEBF16",
    "NK_TARGET_SMEBI32",
    "NK_TARGET_SMELUT2",
    "NK_TARGET_SMEFA64",
    "NK_TARGET_RVV",
    "NK_TARGET_RVVHALF",
    "NK_TARGET_RVVBF16",
    "NK_TARGET_RVVBB",
    "NK_TARGET_V128RELAXED",
    "NK_TARGET_LOONGSONASX",
    "NK_TARGET_POWERVSX",
};
