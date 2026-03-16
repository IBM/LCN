import re
import pyagrum as gum

# 1. Setup a CredalNet step-by-step with interval constraints
cn = gum.CredalNet()
id_x = cn.addVariable("X", 2)  # Binary variable (2 values)
id_y = cn.addVariable("Y", 4)  # 4-value variable
cn.addArc(id_x, id_y)          # X -> Y

# 2. Set interval constraints for X (no parents): [lower, upper] per value
cn.fillConstraints(id_x, [0.2, 0.6], [0.4, 0.8])

# 3. Set interval constraints for Y given X (parent has 2 states)
#    Vectors are concatenated: [Y|X=0, Y|X=1] for both lower and upper
cn.fillConstraints(id_y,
    [0.1, 0.1, 0.2, 0.1, 0.05, 0.2, 0.3, 0.05],  # lower: Y|X=0 then Y|X=1
    [0.3, 0.4, 0.5, 0.3, 0.20, 0.4, 0.5, 0.20])   # upper: Y|X=0 then Y|X=1

# 4. Compute extreme points of each credal set using lrs
cn.intervalToCredal()
print(f"Separately specified: {cn.isSeparatelySpecified()}")

# # 5. Run inference
# ie = gum.CNMonteCarloSampling(cn)
# ie.makeInference()

# print("\nMarginal bounds for X:")
# print(f"  min: {ie.marginalMin('X')}")
# print(f"  max: {ie.marginalMax('X')}")

# print("\nMarginal bounds for Y:")
# print(f"  min: {ie.marginalMin('Y')}")
# print(f"  max: {ie.marginalMax('Y')}")

# 6. Enumerate extreme points of the credal sets for Y
#    Parse the string representation of the CredalNet which lists vertices
#    per node and per parent instantiation.
def parse_credal_vertices(cn, var_name):
    """Parse extreme points from the CredalNet string representation."""
    output = str(cn)
    results = {}
    print(output)
    # Match lines for the target variable: <parent_config> : [[v1, v2, ...], ...]
    in_var = False
    for line in output.splitlines():
        if line.startswith(f"{var_name}:"):
            in_var = True
            continue
        if in_var:
            if line.strip() == "" or (not line.startswith("<")):
                break
            # Extract parent config and vertices
            match = re.match(r"(<[^>]*>)\s*:\s*\[(.+)\]$", line.strip())
            if match:
                parent_cfg = match.group(1)
                # Parse nested list: [[a, b, ...], [c, d, ...], ...]
                inner = match.group(2)
                vertices = []
                for vmatch in re.finditer(r"\[([^\]]+)\]", inner):
                    vertex = [float(x.strip()) for x in vmatch.group(1).split(",")]
                    vertices.append(vertex)
                results[parent_cfg] = vertices
    return results

vertices_y = parse_credal_vertices(cn, "Y")
print(f"\nExtreme points of credal sets for Y:")
for parent_cfg, verts in vertices_y.items():
    print(f"  {parent_cfg}: {len(verts)} vertices")
    for i, v in enumerate(verts):
        v_str = ", ".join(f"{x:.4f}" for x in v)
        print(f"    vertex {i}: [{v_str}]")
