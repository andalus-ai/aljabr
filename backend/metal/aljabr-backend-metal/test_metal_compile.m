#import "aljabr_metal_kernel_source.h"
#import <Foundation/Foundation.h>

int main() {
  NSString *src = aljabr_metal_matvec_kernel_source(256);
  [src writeToFile:@"test.metal"
        atomically:YES
          encoding:NSUTF8StringEncoding
             error:nil];
  return 0;
}
