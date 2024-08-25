import numpy as np

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)

from evalplus.eval._special_oracle import (
    MBPP_OUTPUT_NOT_NONE_TASKS,
    MBPP_OUTPUT_SET_EQ_TASKS,
    _poly,
)

def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False

def ut_exact_match(
    hyp_ut, 
    ref_ut, 
    entry_point, 
    dataset, 
    inp=None, 
    atol=0 # need to change this later
    ):
    exact_match = hyp_ut == ref_ut

    # ================================================ #
    # ============== special oracles ================= #
    if dataset == "mbpp":
        if "are_equivalent" == entry_point:  # Mbpp/164 special oracle
            exact_match = exact_match or True
        elif "sum_div" == entry_point:  # Mbpp/295 special oracle
            exact_match = exact_match or hyp_ut == 0 or ref_ut == 0
        elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
            exact_match = set(hyp_ut) == set(ref_ut)
        elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
            # exp is True  if not None
            #        False if None
            if isinstance(hyp_ut, bool):
                hyp_ut = hyp_ut is not None
            if isinstance(ref_ut, bool):
                ref_ut = ref_ut is not None
            exact_match = hyp_ut == ref_ut

    if dataset == "humaneval":
        if "find_zero" == entry_point:
            hyp_ut = _poly(*inp, hyp_ut) <= atol
            ref_ut = _poly(*inp, ref_ut) <= atol
            exact_match = hyp_ut == ref_ut
    # ============== special oracles ================= #
    # ================================================ #

    if atol == 0 and (is_floats(ref_ut) or is_floats(hyp_ut)):
        atol = 1e-6  # enforce atol for float comparison
    if not exact_match and atol != 0:
        # explicitly set rtol=1e-07
        # to match `np.testing.assert_allclose`'s default values
        exact_match =  np.allclose(hyp_ut, ref_ut, rtol=1e-07, atol=atol)
    
    return int(exact_match)