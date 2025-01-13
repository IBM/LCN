# Debugging the Pearl example

from lcn.model import LCN
from lcn.inference.utils import check_consistency
from lcn.inference.exact_marginal import ExactMarginalInference
from lcn.inference.approx_marginal import ApproximateMarginalInference

if __name__ == "__main__":

    # Load the LCN
    file_name = "examples/pearl_twin_network3.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    # ok = check_consistency(l)
    # if ok:
    #     print("CONSISTENT")
    # else:
    #     print("INCONSISTENT")

    # Run exact marginal inference
    # query = "X3"
    # algo = ExactMarginalInference(lcn=l)
    # algo.run(query_formula=query, debug=True)

    # Run approximate marginal inference
    algo = ApproximateMarginalInference(lcn=l)
    algo.run(n_iters=10, threshold=0.000001, debug=False, max_factors=True)
