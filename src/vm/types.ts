/**
 * VM & Device Types — Prism IDE Integration
 * ==========================================
 * Interfaces for virtual machine management, device templates,
 * and deployment pipelines. Apps deployed via this system inherit
 * Synalux subscription tiers (free → enterprise).
 *
 * Only interfaces live in Prism — implementations stay in Synalux.
 *
 * @see synalux-private/portal/src/lib/vm-manager.ts
 * @see synalux-private/portal/src/lib/device-registry.ts
 */

import type { ScmTier } from '../scm/types';

// ── CPU Architecture ────────────────────────────────────────

export type CpuArch = 'x86_64' | 'arm64' | 'universal';

// ── Hypervisor Support ──────────────────────────────────────

export type HypervisorType =
    | 'apple_virtualization'   // macOS native (Apple Silicon)
    | 'vmware_fusion'          // VMware Fusion / Workstation
    | 'parallels'              // Parallels Desktop
    | 'qemu'                   // QEMU/KVM (Linux)
    | 'hyper_v'                // Windows Hyper-V
    | 'xcode_simulator'        // iOS/watchOS/tvOS simulators
    | 'android_emulator'       // Android AVD
    | 'custom';                // User-provided

// ── OS Platform ─────────────────────────────────────────────

export type OsPlatform =
    | 'linux'
    | 'windows'
    | 'macos'
    | 'ios'
    | 'ipados'
    | 'watchos'
    | 'tvos'
    | 'android'
    | 'wear_os'
    | 'custom';

// ── Device Form Factor ──────────────────────────────────────

export type DeviceFormFactor =
    | 'desktop'
    | 'laptop'
    | 'server'
    | 'phone'
    | 'tablet'
    | 'watch'
    | 'tv'
    | 'embedded'
    | 'custom';

// ── VM Hardware Spec ────────────────────────────────────────

export interface VmHardwareSpec {
    cpu_arch: CpuArch;
    cpu_cores: number;
    ram_gb: number;
    storage_gb: number;
    gpu_enabled: boolean;
    gpu_vram_gb?: number;
    network_mode: 'nat' | 'bridged' | 'host_only' | 'none';
}

// ── VM Image ────────────────────────────────────────────────

export interface VmImage {
    id: string;
    name: string;
    platform: OsPlatform;
    version: string;               // e.g. "24.04 LTS", "11 Pro", "18.0"
    arch: CpuArch;
    form_factor: DeviceFormFactor;
    hypervisor: HypervisorType;
    hardware: VmHardwareSpec;

    /** Preview image URL for the IDE gallery */
    preview_image: string;

    /** Is this a built-in template or user-created? */
    source: 'builtin' | 'imported' | 'custom';

    /** For imported VMs: original file path (.vmx, .pvm, .ova, .qcow2) */
    import_path?: string;
    import_format?: 'vmx' | 'pvm' | 'ova' | 'qcow2' | 'vhdx' | 'raw';

    /** User-defined custom parameters (key-value pairs) */
    custom_params?: Record<string, string | number | boolean>;

    /** Synalux tier this VM's deployed apps inherit */
    inherited_tier: ScmTier;

    created_at: string;
    updated_at: string;
}

// ── Built-in Device Templates ───────────────────────────────

export interface DeviceTemplate {
    id: string;
    name: string;
    description: string;
    platform: OsPlatform;
    form_factor: DeviceFormFactor;
    arch: CpuArch[];              // Supported architectures
    default_hardware: VmHardwareSpec;
    os_versions: string[];        // Available OS versions
    preview_image: string;
    device_variants: string[];    // e.g. ["iPhone 16 Pro", "iPhone 16 Pro Max"]

    /** Min tier required to use this template */
    min_tier: ScmTier;
}

/**
 * Built-in device templates — the IDE gallery.
 * Users can create custom devices via VmImage with source: 'custom'.
 */
export const DEVICE_TEMPLATES: DeviceTemplate[] = [
    // ── Linux ───────────────────────────────────────────
    {
        id: 'linux-ubuntu',
        name: 'Ubuntu Server / Desktop',
        description: 'Ubuntu 24.04 LTS with full dev toolchain. Supports x86_64 and ARM64 natively.',
        platform: 'linux',
        form_factor: 'server',
        arch: ['x86_64', 'arm64'],
        default_hardware: {
            cpu_arch: 'arm64', cpu_cores: 4, ram_gb: 8,
            storage_gb: 64, gpu_enabled: false, network_mode: 'bridged',
        },
        os_versions: ['24.04 LTS', '22.04 LTS', '20.04 LTS'],
        preview_image: '/vm/linux-ubuntu.png',
        device_variants: ['Ubuntu Desktop', 'Ubuntu Server', 'Ubuntu Minimal'],
        min_tier: 'free',
    },
    {
        id: 'linux-fedora',
        name: 'Fedora Workstation',
        description: 'Fedora 41 with GNOME desktop. Ideal for cutting-edge development.',
        platform: 'linux',
        form_factor: 'desktop',
        arch: ['x86_64', 'arm64'],
        default_hardware: {
            cpu_arch: 'arm64', cpu_cores: 4, ram_gb: 8,
            storage_gb: 64, gpu_enabled: false, network_mode: 'bridged',
        },
        os_versions: ['41', '40', '39'],
        preview_image: '/vm/linux-fedora.png',
        device_variants: ['Workstation', 'Server', 'CoreOS'],
        min_tier: 'free',
    },

    // ── Windows ─────────────────────────────────────────
    {
        id: 'windows-11',
        name: 'Windows 11 Pro',
        description: 'Windows 11 with .NET, Visual Studio, and PowerShell. x86_64 and ARM64.',
        platform: 'windows',
        form_factor: 'desktop',
        arch: ['x86_64', 'arm64'],
        default_hardware: {
            cpu_arch: 'x86_64', cpu_cores: 4, ram_gb: 8,
            storage_gb: 128, gpu_enabled: true, gpu_vram_gb: 2, network_mode: 'nat',
        },
        os_versions: ['11 Pro 24H2', '11 Pro 23H2', '10 Pro 22H2'],
        preview_image: '/vm/windows-11.png',
        device_variants: ['Windows 11 Pro', 'Windows 11 Enterprise', 'Windows Server 2025'],
        min_tier: 'standard',
    },

    // ── iOS / iPadOS ────────────────────────────────────
    {
        id: 'ios-simulator',
        name: 'iOS Simulator',
        description: 'Xcode iOS Simulator for iPhone and iPad. SwiftUI live preview included.',
        platform: 'ios',
        form_factor: 'phone',
        arch: ['arm64'],
        default_hardware: {
            cpu_arch: 'arm64', cpu_cores: 2, ram_gb: 4,
            storage_gb: 16, gpu_enabled: true, gpu_vram_gb: 1, network_mode: 'nat',
        },
        os_versions: ['18.0', '17.5', '16.4'],
        preview_image: '/vm/ios-simulator.png',
        device_variants: ['iPhone 16 Pro', 'iPhone 16 Pro Max', 'iPhone 16', 'iPhone SE'],
        min_tier: 'standard',
    },
    {
        id: 'ipados-simulator',
        name: 'iPadOS Simulator',
        description: 'Xcode iPadOS Simulator for iPad Pro and iPad Air.',
        platform: 'ipados',
        form_factor: 'tablet',
        arch: ['arm64'],
        default_hardware: {
            cpu_arch: 'arm64', cpu_cores: 4, ram_gb: 8,
            storage_gb: 32, gpu_enabled: true, gpu_vram_gb: 2, network_mode: 'nat',
        },
        os_versions: ['18.0', '17.5'],
        preview_image: '/vm/ipados-simulator.png',
        device_variants: ['iPad Pro 13"', 'iPad Pro 11"', 'iPad Air', 'iPad mini'],
        min_tier: 'standard',
    },

    // ── watchOS / Wear OS ───────────────────────────────
    {
        id: 'watchos-simulator',
        name: 'Apple Watch Simulator',
        description: 'watchOS simulator for Apple Watch. HealthKit and workout APIs included.',
        platform: 'watchos',
        form_factor: 'watch',
        arch: ['arm64'],
        default_hardware: {
            cpu_arch: 'arm64', cpu_cores: 1, ram_gb: 1,
            storage_gb: 4, gpu_enabled: false, network_mode: 'nat',
        },
        os_versions: ['11.0', '10.5', '10.0'],
        preview_image: '/vm/watchos-simulator.png',
        device_variants: ['Apple Watch Ultra 2', 'Apple Watch Series 10', 'Apple Watch SE'],
        min_tier: 'advanced',
    },
    {
        id: 'wearos-emulator',
        name: 'Wear OS Emulator',
        description: 'Google Wear OS emulator for Samsung Galaxy Watch and Pixel Watch.',
        platform: 'wear_os',
        form_factor: 'watch',
        arch: ['arm64', 'x86_64'],
        default_hardware: {
            cpu_arch: 'arm64', cpu_cores: 1, ram_gb: 1,
            storage_gb: 4, gpu_enabled: false, network_mode: 'nat',
        },
        os_versions: ['5.0', '4.0'],
        preview_image: '/vm/wearos-emulator.png',
        device_variants: ['Galaxy Watch 7', 'Galaxy Watch Ultra', 'Pixel Watch 3'],
        min_tier: 'advanced',
    },

    // ── Android ─────────────────────────────────────────
    {
        id: 'android-emulator',
        name: 'Android Emulator',
        description: 'Android 15 emulator with Material Design 3. Pixel and Samsung skins.',
        platform: 'android',
        form_factor: 'phone',
        arch: ['arm64', 'x86_64'],
        default_hardware: {
            cpu_arch: 'arm64', cpu_cores: 4, ram_gb: 4,
            storage_gb: 32, gpu_enabled: true, gpu_vram_gb: 1, network_mode: 'nat',
        },
        os_versions: ['15 (API 35)', '14 (API 34)', '13 (API 33)'],
        preview_image: '/vm/android-emulator.png',
        device_variants: ['Pixel 9 Pro', 'Pixel 9', 'Samsung Galaxy S25 Ultra', 'Samsung Galaxy S25'],
        min_tier: 'standard',
    },
    {
        id: 'android-tablet',
        name: 'Android Tablet Emulator',
        description: 'Android tablet emulator for Samsung Galaxy Tab and Pixel Tablet.',
        platform: 'android',
        form_factor: 'tablet',
        arch: ['arm64', 'x86_64'],
        default_hardware: {
            cpu_arch: 'arm64', cpu_cores: 4, ram_gb: 8,
            storage_gb: 64, gpu_enabled: true, gpu_vram_gb: 2, network_mode: 'nat',
        },
        os_versions: ['15 (API 35)', '14 (API 34)'],
        preview_image: '/vm/android-tablet.png',
        device_variants: ['Galaxy Tab S10 Ultra', 'Pixel Tablet'],
        min_tier: 'advanced',
    },
];

// ── VM Import ───────────────────────────────────────────────

export type VmImportFormat = 'vmx' | 'pvm' | 'ova' | 'qcow2' | 'vhdx' | 'raw';

export interface VmImportRequest {
    /** Path to the VM image file on disk */
    source_path: string;
    format: VmImportFormat;
    /** User-defined name for the imported VM */
    name: string;
    /** Override default hardware spec */
    hardware_overrides?: Partial<VmHardwareSpec>;
    /** User-supplied custom parameters (unlimited key-value) */
    custom_params?: Record<string, string | number | boolean>;
    /** Tier for apps deployed from this VM */
    inherited_tier?: ScmTier;
}

export interface VmImportResult {
    success: boolean;
    vm_id: string;
    image: VmImage;
    warnings: string[];
    import_time_ms: number;
}

// ── Custom Device Parameters ────────────────────────────────

export interface CustomDeviceSpec {
    name: string;
    platform: OsPlatform;
    form_factor: DeviceFormFactor;
    arch: CpuArch;
    os_version: string;
    hardware: VmHardwareSpec;
    hypervisor: HypervisorType;

    /** Screen resolution for device simulators */
    screen_resolution?: { width: number; height: number; dpi: number };

    /** User-defined parameters — fully open-ended */
    custom_params: Record<string, string | number | boolean>;

    /** Environment variables to inject into the VM */
    env_vars?: Record<string, string>;

    /** Startup script to run on boot */
    startup_script?: string;

    /** Port forwarding rules */
    port_forwards?: Array<{ host: number; guest: number; protocol: 'tcp' | 'udp' }>;
}

// ── Deployment ──────────────────────────────────────────────

export type DeployTarget =
    | 'github_actions'
    | 'vercel'
    | 'local_vm'
    | 'synalux_cloud'
    | 'custom_server';

export interface DeployConfig {
    target: DeployTarget;
    vm_id?: string;                // Deploy to a specific VM
    repo_url?: string;             // GitHub repo for CI/CD
    branch: string;
    build_command: string;
    test_command?: string;
    env_vars: Record<string, string>;

    /** Apps inherit Synalux tier from the workspace */
    inherited_tier: ScmTier;

    /** Auto-deploy on push? */
    auto_deploy: boolean;
}

export interface DeployResult {
    success: boolean;
    deploy_id: string;
    url?: string;
    logs_url?: string;
    test_results?: {
        passed: number;
        failed: number;
        skipped: number;
        duration_ms: number;
    };
    timestamp: string;
}

// ── VM Tier Limits ──────────────────────────────────────────

export interface VmTierLimits {
    /** Max concurrent VMs */
    max_concurrent_vms: number;
    /** Max RAM allocated across all VMs (GB) */
    max_total_ram_gb: number;
    /** Max storage across all VMs (GB) */
    max_total_storage_gb: number;
    /** Available OS platforms */
    platforms: OsPlatform[];
    /** Can import from VMware/Parallels? */
    vm_import: boolean;
    /** Can create custom devices? */
    custom_devices: boolean;
    /** GPU passthrough support */
    gpu_passthrough: boolean;
    /** Max deployments per day */
    deploys_per_day: number;
    /** Thin-client proxy to Synalux */
    thin_client: boolean;
}

export const VM_TIERS: Record<ScmTier, VmTierLimits> = {
    free: {
        max_concurrent_vms: 1,
        max_total_ram_gb: 4,
        max_total_storage_gb: 32,
        platforms: ['linux'],
        vm_import: false,
        custom_devices: false,
        gpu_passthrough: false,
        deploys_per_day: 3,
        thin_client: false,
    },
    standard: {
        max_concurrent_vms: 3,
        max_total_ram_gb: 16,
        max_total_storage_gb: 256,
        platforms: ['linux', 'windows', 'ios', 'android'],
        vm_import: true,
        custom_devices: true,
        gpu_passthrough: false,
        deploys_per_day: 25,
        thin_client: true,
    },
    advanced: {
        max_concurrent_vms: 8,
        max_total_ram_gb: 64,
        max_total_storage_gb: 1024,
        platforms: ['linux', 'windows', 'macos', 'ios', 'ipados', 'android', 'watchos', 'wear_os'],
        vm_import: true,
        custom_devices: true,
        gpu_passthrough: true,
        deploys_per_day: 100,
        thin_client: true,
    },
    enterprise: {
        max_concurrent_vms: Infinity,
        max_total_ram_gb: Infinity,
        max_total_storage_gb: Infinity,
        platforms: ['linux', 'windows', 'macos', 'ios', 'ipados', 'android', 'watchos', 'wear_os', 'tvos', 'custom'],
        vm_import: true,
        custom_devices: true,
        gpu_passthrough: true,
        deploys_per_day: Infinity,
        thin_client: true,
    },
};
