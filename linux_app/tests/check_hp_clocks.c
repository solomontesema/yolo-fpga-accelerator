/**
 * Check HP Port Clock Status on ZynqMP
 * 
 * This reads the CRF_APB registers to verify HP port clocks are enabled.
 * On ZynqMP, the HP port clocks are controlled by CRF_APB registers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// ZynqMP Clock Register Addresses
#define CRF_APB_BASE        0xFD1A0000UL
#define CRL_APB_BASE        0xFF5E0000UL
#define CRF_APB_SIZE        0x1000

// HP Port related registers in CRF_APB
// These control the FPD (Full Power Domain) clocks
#define CRF_APB_TOPSW_MAIN_CTRL     0x00C4  // Top switch main clock
#define CRF_APB_TOPSW_LSBUS_CTRL    0x00C8  // Top switch LS bus clock
#define CRF_APB_DBG_FPD_CTRL        0x00A0  // FPD debug clock
#define CRF_APB_DP_VIDEO_CTRL       0x0070  // DP video clock
#define CRF_APB_DP_AUDIO_CTRL       0x0074  // DP audio clock
#define CRF_APB_DP_STC_CTRL         0x0078  // DP STC clock
#define CRF_APB_DDR_CTRL            0x0080  // DDR clock

// AFI (AXI FIFO Interface) control - these control HP/HPC ports
// Located in LPD_SLCR
#define LPD_SLCR_BASE       0xFF410000UL
#define AFI_FS              0x5000  // AFI Full-speed control
#define AFI_FM0             0x5008  // AFI FM0 (HP0)
#define AFI_FM1             0x500C  // AFI FM1 (HP1)  
#define AFI_FM2             0x5010  // AFI FM2 (HP2)
#define AFI_FM3             0x5014  // AFI FM3 (HP3)
#define AFI_FM4             0x5018  // AFI FM4 (HPC0)
#define AFI_FM5             0x501C  // AFI FM5 (HPC1)

// FPD_SLCR registers for HP port config
#define FPD_SLCR_BASE       0xFD610000UL
#define FPD_SLCR_SIZE       0x2000
#define AFI_FS_FPD          0x0A00  // AFI FS in FPD_SLCR

int main(void) {
    int fd;
    volatile uint32_t *crf_apb, *lpd_slcr, *fpd_slcr;
    
    printf("=============================================\n");
    printf("ZynqMP HP Port Clock/Config Check\n");
    printf("=============================================\n\n");
    
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("Cannot open /dev/mem");
        return 1;
    }
    
    // Map CRF_APB
    crf_apb = mmap(NULL, CRF_APB_SIZE, PROT_READ, MAP_SHARED, fd, CRF_APB_BASE);
    if (crf_apb == MAP_FAILED) {
        perror("Cannot mmap CRF_APB");
        close(fd);
        return 1;
    }
    
    printf("[1] CRF_APB Clock Control Registers:\n");
    printf("    TOPSW_MAIN_CTRL  (0x%04x): 0x%08x\n", CRF_APB_TOPSW_MAIN_CTRL, 
           crf_apb[CRF_APB_TOPSW_MAIN_CTRL / 4]);
    printf("    TOPSW_LSBUS_CTRL (0x%04x): 0x%08x\n", CRF_APB_TOPSW_LSBUS_CTRL,
           crf_apb[CRF_APB_TOPSW_LSBUS_CTRL / 4]);
    printf("    DDR_CTRL         (0x%04x): 0x%08x\n", CRF_APB_DDR_CTRL,
           crf_apb[CRF_APB_DDR_CTRL / 4]);
    
    munmap((void*)crf_apb, CRF_APB_SIZE);
    
    // Map FPD_SLCR  
    fpd_slcr = mmap(NULL, FPD_SLCR_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, FPD_SLCR_BASE);
    if (fpd_slcr == MAP_FAILED) {
        perror("Cannot mmap FPD_SLCR");
        close(fd);
        return 1;
    }
    
    printf("\n[2] FPD_SLCR AFI (HP Port) Registers:\n");
    
    // Read AFI registers - these control HP port configuration
    // AFI_FS controls whether the HP ports use full-speed or low-speed interface
    uint32_t afi_fs = fpd_slcr[AFI_FS_FPD / 4];
    printf("    AFI_FS (0x%04x): 0x%08x\n", AFI_FS_FPD, afi_fs);
    
    // Check width settings (important for 64-bit vs 128-bit)
    printf("      HP0 width: %s\n", (afi_fs & 0x100) ? "128-bit" : "64-bit");
    printf("      HP1 width: %s\n", (afi_fs & 0x200) ? "128-bit" : "64-bit");
    printf("      HP2 width: %s\n", (afi_fs & 0x400) ? "128-bit" : "64-bit");
    printf("      HP3 width: %s\n", (afi_fs & 0x800) ? "128-bit" : "64-bit");
    
    // Read more AFI config registers
    printf("\n[3] Checking additional FPD_SLCR registers:\n");
    
    // Offset 0x0000 - wprot0
    printf("    FPD_SLCR_WPROT0 (0x0000): 0x%08x\n", fpd_slcr[0]);
    
    // Offset 0x0044 - Interconnect control
    printf("    INTER_CTRL (0x0044): 0x%08x\n", fpd_slcr[0x44 / 4]);
    
    munmap((void*)fpd_slcr, FPD_SLCR_SIZE);
    
    // Try to check if DDR controller is accessible
    printf("\n[4] DDR Controller Check:\n");
    volatile uint32_t *ddr_ctrl = mmap(NULL, 0x1000, PROT_READ, MAP_SHARED, fd, 0xFD070000UL);
    if (ddr_ctrl != MAP_FAILED) {
        printf("    DDR_MSTR (0x000): 0x%08x\n", ddr_ctrl[0]);
        printf("    DDR_STAT (0x004): 0x%08x\n", ddr_ctrl[1]);
        
        uint32_t stat = ddr_ctrl[1];
        printf("    Operating mode: %d\n", stat & 0x7);
        printf("    Self-refresh: %s\n", (stat & 0x10) ? "yes" : "no");
        
        munmap((void*)ddr_ctrl, 0x1000);
    } else {
        printf("    Cannot read DDR controller\n");
    }
    
    // Check PS-PL AXI interface status
    printf("\n[5] PS Version / Silicon Info:\n");
    volatile uint32_t *csu = mmap(NULL, 0x1000, PROT_READ, MAP_SHARED, fd, 0xFFCA0000UL);
    if (csu != MAP_FAILED) {
        printf("    CSU_VERSION: 0x%08x\n", csu[0x44 / 4]);
        printf("    IDCODE: 0x%08x\n", csu[0x40 / 4]);
        munmap((void*)csu, 0x1000);
    }
    
    close(fd);
    
    printf("\n=============================================\n");
    printf("If HP ports are configured correctly, the AFI_FS\n");
    printf("register should show appropriate width settings.\n");
    printf("\n");
    printf("The accelerator timeout suggests the AXI master\n");
    printf("cannot complete transactions to DDR. This could be:\n");
    printf("  1. SmartConnect clock not running\n");
    printf("  2. HP port AXI interface not enabled\n");
    printf("  3. Address translation issue\n");
    printf("=============================================\n");
    
    return 0;
}
