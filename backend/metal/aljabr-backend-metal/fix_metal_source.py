import re

with open("src/main/cpp/metal/aljabr_metal_kernel_source.m", "r") as f:
    code = f.read()

start_idx = code.find("NSString* aljabr_metal_matvec_kernel_source(NSUInteger threads)")
if start_idx != -1:
    prefix = code[:start_idx]
    suffix = code[start_idx:]
    
    # We want to replace % with %% EXCEPT for %lu.
    # We should match `%` that is not followed by `%` and not followed by `lu`.
    suffix = re.sub(r'%(?!lu|%)', '%%', suffix)
    
    with open("src/main/cpp/metal/aljabr_metal_kernel_source.m", "w") as f:
        f.write(prefix + suffix)
    print("Fixed!")
else:
    print("Not found")

