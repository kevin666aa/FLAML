import os
from flaml import oai
from flaml.autogen.math.math_voting import SelfConsistency
from flaml.autogen.math.math_solver import MathSolver
import argparse
from utils import mylogger, load_level5_math_each_category

def parse_args():
    parser = argparse.ArgumentParser(description="Math Solver")
    parser.add_argument("--prompt_type", "-ptype", dest="prompt_type", help="prompt type", default="select", type=str)
    parser.add_argument("--prompt_location", dest="prompt_location", help="prompt location", default="user", type=str)
    parser.add_argument("--max_round", dest="max_round", help="max round", default=15, type=int)
    parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./autotools", type=str)
    parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache", help="cache folder")
    parser.add_argument("--samples_per_category", help="samples per category", default=20, type=int)
    parser.add_argument("--temperature", "-t", dest="temperature", help="temperature", default=1, type=float)
    parser.add_argument("--test_run", help="test run", action="store_true")
    parser.add_argument("--categories", dest="categories", help="categories", default=[0, 1], nargs="+")

    # not used

    parser.add_argument("--n", dest="n", help="number of samples", default=1, type=int)
    parser.add_argument("--voting", action="store_true")
    args = parser.parse_args()
    args.folder = args.folder + "_" + args.prompt_location + "_" + args.prompt_type + "_t" + str(args.temperature)
    os.makedirs(args.folder, exist_ok=True)
    return args

def pseudo_main():

    # 2. args, settings and logger
    args = parse_args()
    args.model = "gpt-4"
    oai.ChatCompletion.request_timeout = 60 * 10  # 10 minutes
    oai.ChatCompletion.set_cache(seed=41, cache_path=args.cache_folder)
    logger = mylogger(os.path.join(args.folder, "log.txt"))

    # 3. load math dataset
    problem_sets = load_level5_math_each_category(
        samples_per_category=args.samples_per_category, category_to_load=args.categories
    )
    if args.test_run:
        problem_sets = load_level5_math_each_category(samples_per_category=1, category_to_load=args.categories)
        logger.log("Take out 1 problem from each category for test run.")

    # 4. solve
    if not args.voting:
        solver = MathSolver(
            model=args.model,
            prompt_type=args.prompt_type,
            max_round=args.max_round,
            temperature=args.temperature,
            prompt_location=args.prompt_location,
            logger=logger,
        )
        with open(os.path.join(args.folder, "prompt.txt"), "w") as f:
            f.write(solver.prompt)

        for problem_set in problem_sets:
            for i in range(len(problem_set)):
                problem_set[i]["problem_id"] = str(i)  # assign problem id

            solver.solve_one_category(problem_set, saving_folder=args.folder)
            os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

        logger.log("****************************\n\n\n\n\n", verbose=False)
        os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

    else:
        logger.log("Voting is not supported yet.")
        pass

    # problem_sets = load_level5_math_each_category()
    # for problem_set in problem_sets:
    #     for i in range(len(problem_set)):
    #         problem_set[i]['problem_id'] = str(i)

    #     print('Take out 2 problems from each category for testing.')
    #     problem_set = problem_set[:1] # test with only 2 problems first
    #     # vanilla_voting_one_category(model, problem_set, saving_folder='./voting')
    #     break
