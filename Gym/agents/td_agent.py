#!/usr/bin/env python
import os
import sys
import random
import time
import logging
import json
from collections import defaultdict
from itertools import product
from multiprocessing import Pool
from tempfile import NamedTemporaryFile

import pandas as pd
import click
from tqdm import tqdm as _tqdm  # 진행상황 표시 bar 출력
tqdm = _tqdm

from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD
from examples.human_agent import HumanAgent
from examples.base_agent import BaseAgent


DEFAULT_VALUE = 0
EPISODE_CNT = 17000
BENCH_EPISODE_CNT = 3000
MODEL_FILE = 'td_agent.dat'
EPSILON = 0.08
ALPHA = 0.4
CWD = os.path.dirname(os.path.abspath(__file__))


st_values = {}
st_visits = defaultdict(lambda: 0)


def reset_state_values():  # 상태 초기화
    global st_values, st_visits
    st_values = {}
    st_visits = defaultdict(lambda: 0)


def set_state_value(state, value):  # 상태 설정
    st_visits[state] += 1
    st_values[state] = value


def best_val_indices(values, fn):  # values를 함께 들어온 함수 fn에 주고 value가 최적인 값들을 list로 만들어 return
    best = fn(values)
    return [i for i, v in enumerate(values) if v == best]


class TDAgent(object):
    def __init__(self, mark, epsilon, alpha):  # 초기화
        self.mark = mark
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode_rate = 1.0

    def act(self, state, ava_actions):
        return self.egreedy_policy(state, ava_actions)

    def egreedy_policy(self, state, ava_actions):  # epsilon greedy에 따라 취할 수 있는 action들 중 하나를 택해 return한다.
        """Returns action by Epsilon greedy policy.

        Return random action with epsilon probability or best action.

        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions

        Returns:
            int: Selected action.
        """
        logging.debug("egreedy_policy for '{}'".format(self.mark))
        e = random.random()
        if e < self.epsilon * self.episode_rate:
            logging.debug("Explore with eps {}".format(self.epsilon))
            action = self.random_action(ava_actions)
        else:
            logging.debug("Exploit with eps {}".format(self.epsilon))
            action = self.greedy_action(state, ava_actions)
        return action

    def random_action(self, ava_actions):  # 취할 수 있는 action들 중 무작위로 하나를 택한다.
        return random.choice(ava_actions)

    def greedy_action(self, state, ava_actions):  # 현재 상태에서 가장 greedy한(보상을 크게 만드는) action을 취한다.
        """Return best action by current state value.

        Evaluate each action, select best one. Tie-breaking is random.

        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions

        Returns:
            int: Selected action
        """
        assert len(ava_actions) > 0

        ava_values = []
        for action in ava_actions:
            nstate = after_action_state(state, action)
            nval = self.ask_value(nstate)
            ava_values.append(nval)
            vcnt = st_visits[nstate]
            logging.debug("  nstate {} val {:0.2f} visits {}".
                          format(nstate, nval, vcnt))

        # select most right action for 'O' or 'X'
        if self.mark == 'O':
            indices = best_val_indices(ava_values, max)
        else:
            indices = best_val_indices(ava_values, min)

        # tie breaking by random choice
        aidx = random.choice(indices)
        logging.debug("greedy_action mark {} ava_values {} indices {} aidx {}".
                      format(self.mark, ava_values, indices, aidx))

        action = ava_actions[aidx]

        return action

    def ask_value(self, state):  # 주어진 state의 value를 return, reward에 따라 value의 갱신도 일어난다.
        """Returns value of given state.

        If state is not exists, set it as default value.

        Args:
            state (tuple): State.

        Returns:
            float: Value of a state.
        """
        if state not in st_values:
            logging.debug("ask_value - new state {}".format(state))
            gstatus = check_game_status(state[0])
            val = DEFAULT_VALUE
            # win
            if gstatus > 0:
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return st_values[state]

    def backup(self, state, nstate, reward):  # action을 취한 후 next state의 value 중 가장 좋은 value로 update
        """Backup value by difference and step size.

        Execute an action then backup Q by best value of next state.

        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action
        """
        logging.debug("backup state {} nstate {} reward {}".
                      format(state, nstate, reward))

        val = self.ask_value(state)
        nval = self.ask_value(nstate)
        diff = nval - val
        val2 = val + self.alpha * diff  # alpha = 학습률, 다음 기대 가치와 현재 가치의 차이를 alpha만큼 갱신(backup)

        logging.debug("  value from {:0.2f} to {:0.2f}".format(val, val2))
        set_state_value(state, val2)


@click.group()
@click.option('-v', '--verbose', count=True, help="Increase verbosity.")
@click.pass_context
def cli(ctx, verbose):  # ? CLI 환경에 진행상황(tdqm)을 띄우나봄. verbose?
    global tqdm

    set_log_level_by(verbose)
    if verbose > 0:
        tqdm = lambda x: x  # NOQA


@cli.command(help="Learn and save the model.")
@click.option('-p', '--episode', "max_episode", default=EPISODE_CNT,
              show_default=True, help="Episode count.")
@click.option('-e', '--epsilon', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-a', '--alpha', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--save-file', default=MODEL_FILE, show_default=True,
              help="Save model data as file name.")
def learn(max_episode, epsilon, alpha, save_file):  # learn option이다. td_agent.py learn을 입력하면 실행하는 모드
    _learn(max_episode, epsilon, alpha, save_file)


def _learn(max_episode, epsilon, alpha, save_file):  
    """Learn by episodes.

    Make two TD agent, and repeat self play for given episode count.  # TD agent를 두 개 만들고 주어진 episode 수만큼 둘이 알아서 play를 한다.
    Update state values as reward coming from the environment.

    Args:
        max_episode (int): Episode count.
        epsilon (float): Probability of exploration.
        alpha (float): Step size.
        save_file: File name to save result.
    """
    reset_state_values()

    env = TicTacToeEnv()
    agents = [TDAgent('O', epsilon, alpha),
              TDAgent('X', epsilon, alpha)]

    start_mark = 'O'  # 1회 play 전(최초) setting
    for i in tqdm(range(max_episode)):  # max_episode만큼 반복할 것임
        episode = i + 1
        env.show_episode(False, episode)  # 몇 번째 episode인지 출력

        # reset agent for new episode
        for agent in agents:
            agent.episode_rate = episode / float(max_episode)  # 각 agent가 episode를 얼마나 진행했는지 기록

        env.set_start_mark(start_mark)  # 각 play마다 play 환경 reset
        state = env.reset()
        _, mark = state
        done = False
        while not done:  # done이 False이면 게임이 끝나지 않은 것이므로 False인 동안 진행
            agent = agent_by_mark(agents, mark)  # 올바른 차례의 agent를 return, 각 turn마다 설정
            ava_actions = env.available_actions()
            env.show_turn(False, mark)
            action = agent.act(state, ava_actions)

            # update (no rendering)
            nstate, reward, done, info = env.step(action)  # action 수행 후 update
            agent.backup(state, nstate, reward)

            if done:  # play가 완료된 경우 step이 return한 done은 True
                env.show_result(False, mark, reward)  # 결과 출력
                # set terminal state value
                set_state_value(state, reward)

            _, mark = state = nstate  # state를 next state로 갱신

        # rotate start
        start_mark = next_mark(start_mark)  # 다음 play의 시작 mark를 갱신

    # save states
    save_model(save_file, max_episode, epsilon, alpha)  # max_episode만큼 학습한 model 저장


def save_model(save_file, max_episode, epsilon, alpha):  # 학습한 model을 저장한다. episode 수는 몇 개인지, epsilon은 얼마인지, alpha는 얼마인지의 정보도 준다.
    with open(save_file, 'wt') as f:
        # write model info
        info = dict(type="td", max_episode=max_episode, epsilon=epsilon,
                    alpha=alpha)
        # write state values
        f.write('{}\n'.format(json.dumps(info)))
        for state, value in st_values.items():
            vcnt = st_visits[state]
            f.write('{}\t{:0.3f}\t{}\n'.format(state, value, vcnt))


def load_model(filename):  # 저장한 model을 물러온다.
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            val = eval(elms[1])
            vcnt = eval(elms[2])
            st_values[state] = val
            st_visits[state] = vcnt
    return info


@cli.command(help="Play with human.")
@click.option('-f', '--load-file', default=MODEL_FILE, show_default=True,
              help="Load file name.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number when play.")
def play(load_file, show_number):  # CLI에서 play option을 주었을 때 실행한다.
    _play(load_file, HumanAgent('O'), show_number)


def _play(load_file, vs_agent, show_number):  # 학습한 model과 play를 진행한다.
    """Play with learned model.

    Make TD agent and adversarial agnet to play with.  # agent 오타났다.
    Play and switch starting mark when the game finished.
    TD agent behave no exploring action while in play mode.  # ? play 중엔 학습한 model을 벗어나는 실험은 하지 않는다는 뜻인가?

    Args:
        load_file (str):
        vs_agent (object): Enemy agent of TD agent.
        show_number (bool): Whether show grid number for visual hint.
    """
    load_model(load_file)  # 학습한 model을 load
    env = TicTacToeEnv(show_number=show_number)
    td_agent = TDAgent('X', 0, 0)  # prevent exploring  # mark가 X이고 epsilon과 alpha가 각각 0인 agent 생성(컴퓨터 입장에선 이 녀석이 본인이고 vs_agent가 적임. player는 vs_agent)
    start_mark = 'O'
    agents = [vs_agent, td_agent]

    while True:  # 게임은 끝나지 않는단다^^
        # start agent rotation
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False

        # show start board for human agent
        if mark == 'O':
            env.render(mode='human')  # 화면 출력

        while not done:
            agent = agent_by_mark(agents, mark)
            human = isinstance(agent, HumanAgent)  # agent가 HumanAgent인지 알아본다.(bool)

            env.show_turn(True, mark)
            ava_actions = env.available_actions()
            if human:
                action = agent.act(ava_actions)  # HumanAgent의 act
                if action is None:
                    sys.exit()
            else:
                action = agent.act(state, ava_actions)  # TDAgent의 act

            state, reward, done, info = env.step(action)  # 실행항 action에 따라 value들을 갱신

            env.render(mode='human')
            if done:  # 이 판이 끝났으면 결과 출력하고 break(다음 판 실행할 것임)
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state

        # rotation start
        start_mark = next_mark(start_mark)  # 다음 mark부터 시작하도록 setting


@cli.command(help="Learn and benchmark.")
@click.option('-p', '--learn-episode', "max_episode", default=EPISODE_CNT,
              show_default=True, help="Learn episode count.")
@click.option('-b', '--bench-episode', "max_bench_episode",
              default=BENCH_EPISODE_CNT, show_default=True, help="Bench "
              "episode count.")
@click.option('-e', '--epsilon', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-a', '--alpha', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--model-file', default=MODEL_FILE, show_default=True,
              help="Model data file name.")
def learnbench(max_episode, max_bench_episode, epsilon, alpha, model_file):  # CLI option learnbench가 들어오면 실행한다.
    _learnbench(max_episode, max_bench_episode, epsilon, alpha, model_file)


def _learnbench(max_episode, max_bench_episode, epsilon, alpha, model_file,  # show가 True이면 학습을 수행한 후 학습한 model을 benchmarking한다.
                show=True):
    if show:
        print("Learning...")
    _learn(max_episode, epsilon, alpha, model_file)
    if show:
        print("Benchmarking...")
    return _bench(max_bench_episode, model_file, show)


@cli.command(help="Benchmark agent with base agent.")
@click.option('-p', '--episode', "max_episode", default=BENCH_EPISODE_CNT,
              show_default=True, help="Episode count.")
@click.option('-f', '--model-file', default=MODEL_FILE, show_default=True,
              help="Model data file name.")
def bench(model_file, max_episode):  # CLI option bench가 들어오면 실행한다.
    _bench(max_episode, model_file)


def _bench(max_episode, model_file, show_result=True):  # 사람이 테스트하지 않고 자동으로 play하여 성능을 비교할 수 있는 코드이다. 학습한 agent로 게임을 하는 것이 아니라 base agent를 사용하여 play하여 사람이 하는 것과 유사한 동작을 한다. base agent로 많은 play를 한 후 평가한다.
    """Benchmark given model.

    Args:
        max_episode (int): Episode count to benchmark.
        model_file (str): Learned model file name to benchmark.
        show_result (bool): Output result to stdout.

    Returns:
        (dict): Benchmark result.
    """
    minfo = load_model(model_file)
    agents = [BaseAgent('O'), TDAgent('X', 0, 0)]  # base agent는 두어서 이기는 위치가 있으면 그 곳에 두고 아니면 random한 위치에 두는 최소한의 가이드라인 역할만 하는 agent이다.
    # td agent는 학습한 강화학습 agent이다. td는 시간차라는 뜻이다.
    show = False
	
    start_mark = 'O'
    env = TicTacToeEnv()
    env.set_start_mark(start_mark)

    episode = 0
    results = []
    for i in tqdm(range(max_episode)):
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)  # base_agent의 act
            state, reward, done, info = env.step(action)
            if show:
                env.show_turn(True, mark)
                env.render(mode='human')

            if done:
                if show:
                    env.show_result(True, mark, reward)
                results.append(reward)
                break
            else:
                _, mark = state

        # rotation start
        start_mark = next_mark(start_mark)
        episode += 1

    o_win = results.count(1)
    x_win = results.count(-1)
    draw = len(results) - o_win - x_win
    mfile = model_file.replace(CWD + os.sep, '')
    minfo.update(dict(base_win=o_win, td_win=x_win, draw=draw,
                      model_file=mfile))
    result = json.dumps(minfo)

    if show_result:
        print(result)
    return result


@cli.command(help="Learn and play with human.")
@click.option('-p', '--episode', "max_episode", default=EPISODE_CNT,
              show_default=True, help="Episode count.")
@click.option('-e', '--epsilon', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-a', '--alpha', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--model-file', default=MODEL_FILE, show_default=True,
              help="Model file name.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number when play.")
def learnplay(max_episode, epsilon, alpha, model_file, show_number):  # CLI option으로 learnplay가 들어오면 learn 하고 play한다.
    _learn(max_episode, epsilon, alpha, model_file)
    _play(model_file, HumanAgent('O'), show_number)


@cli.command(help="Grid search hyper-parameters.")
@click.option('-q', '--quality', type=click.Choice(['high', 'mid', 'low']),
              default='mid', show_default=True, help="Grid search"
              " quality.")
@click.option('-r', '--reproduce-test', "rtest_cnt", default=3,
              show_default=True, help="Reproducibility test count.")
def gridsearch(quality, rtest_cnt):  # ? CLI option으로 gridsearch가 들어오면 최적의 hyper-parameters를 찾는다. 근데 hyper-parameter가 정확히 어떻게 나오더라
	# hyper-parameter의 다양한 조합을 구성한다. 각 조합을 동시에 병렬로 benchmarking하여 가장 우수한 성능의 조합을 알려준다.
    """Find and output best hyper-parameters.

    Grid search consists of two phase:
        1. Select best 10 candidates of parameter combination.
        1. Carry out reproduce test and output top 5 parameters.

    Args:
        quality (str): Select preset of parameter combination. High quality
            means more granularity in parameter space.
        rtest_cnt (int): Reproduce test count
    """
    st = time.time()
    _gridsearch_candidate(quality)
    _gridsearch_reproduce(rtest_cnt)
    print("Finished in {:0.2f} seconds".format(time.time() - st))


def _gridsearch_reproduce(rtest_cnt):
    """Refine parameter combination by reproduce test, and output best 5.

    Reproduce test is a learn & bench process from each parameter combination.

    1. Select top 10 parameters from previous step.
    2. Execute reproduce test.
    3. Sort by lose rate.
    4. Output best 5 parameters.

    Args:
        rtest_cnt (int): Reproduce test count

    Todo:
        Apply multiprocessor worker
    """
    print("Reproducibility test.")
    with open(os.path.join(CWD, 'gsmodels/result.json'), 'rt') as fr:
        df = pd.DataFrame([json.loads(line) for line in fr])
        top10_df = df.sort_values(['base_win', 'max_episode'])[:10]

    index = []
    vals = []
    # for each candidate
    pbar = _tqdm(total=len(top10_df) * rtest_cnt)
    for idx, row in top10_df.iterrows():
        index.append(idx)
        base_win_sum = 0
        total_play = 0
        # bench repeatedly
        for i in range(rtest_cnt):
            pbar.update()
            learn_episode = row.max_episode
            epsilon = row.epsilon
            alpha = row.alpha
            with NamedTemporaryFile() as tmp:
                res = _learnbench(learn_episode, BENCH_EPISODE_CNT, epsilon,
                                  alpha, tmp.name, False)
            res = json.loads(res)
            base_win_sum += res['base_win']
            total_play += BENCH_EPISODE_CNT
        lose_pct = float(base_win_sum) / rtest_cnt / total_play * 100
        vals.append(round(lose_pct, 2))

    top10_df['lose_pct'] = pd.Series(vals, index=index)

    df = top10_df.sort_values(['lose_pct', 'max_episode']).reset_index()[:5]
    print(df[['lose_pct', 'max_episode', 'alpha', 'epsilon', 'model_file']])


def _gridsearch_candidate(quality):
    """Select best hyper-parameter candiadates by grid search.

    1. Generate parameter combination by product each parameter space.
    2. Spawn processors to learn & bench each combination.
    3. Wait and write result to a file.

    Args:
        quality (str): Choice among 'high', 'mid', 'low'

    Todo:
        Progress bar estimation is not even.
    """
    # disable sub-process's progressbar
    global tqdm
    tqdm = lambda x: x  # NOQA

    if quality == 'high':
        # high
        epsilons = [e * 0.01 for e in range(8, 25, 2)]
        alphas = [a * 0.1 for a in range(2, 8)]
        episodes = [e for e in range(8000, 31000, 3000)]
    elif quality == 'mid':
        # mid
        epsilons = [e * 0.01 for e in range(10, 20, 5)]
        alphas = [a * 0.1 for a in range(3, 7)]
        episodes = [e for e in range(10000, 30000, 5000)]
    else:
        # low
        epsilons = [e * 0.01 for e in range(9, 13, 2)]
        alphas = [a * 0.1 for a in range(4, 6)]
        episodes = [e for e in range(1000, 2000, 300)]

    alphas = [round(a, 2) for a in alphas]
    _args = list(product(episodes, epsilons, alphas))
    args = []
    for i, arg in enumerate(_args):
        arg = list(arg)
        arg.insert(1, BENCH_EPISODE_CNT)  # bench episode count
        arg.append(os.path.join(CWD, 'gsmodels/model_{:03d}.dat'.format(i)))
        arg.append(False)  # supress print
        args.append(arg)  # model file name
    prev_left = total = len(args)

    print("Grid search for {} parameter combinations.".format(total))
    pbar = _tqdm(total=total)
    pool = Pool()
    result = pool.starmap_async(_learnbench, args)
    while True:
        if result.ready():
            break
        if prev_left != result._number_left:
            ucnt = prev_left - result._number_left
            pbar.update(ucnt)
            prev_left = result._number_left
        time.sleep(1)

    ucnt = prev_left - result._number_left
    pbar.update(ucnt)
    pbar.close()

    with open(os.path.join(CWD, 'gsmodels/result.json'), 'wt') as f:
        for r in result.get():
            f.write('{}\n'.format(r))


if __name__ == '__main__':
    cli()
