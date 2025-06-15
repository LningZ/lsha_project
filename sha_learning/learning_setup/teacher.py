import configparser
import os
from typing import List, Dict

import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


from sha_learning.domain.lshafeatures import TimedTrace, FlowCondition, ProbDistribution, Trace
from sha_learning.domain.obstable import ObsTable, Row, State
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp
from sha_learning.domain.sulfeatures import SystemUnderLearning
from sha_learning.learning_setup.fastddtw import fast_ddtw, plot_aligned_signals
from sha_learning.learning_setup.logger import Logger
from sha_learning.learning_setup.trace_gen import TraceGenerator

LOGGER = Logger('TEACHER')

config = configparser.ConfigParser()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']
NOISE = float(config['LSHA PARAMETERS']['DELTA'])
P_VALUE = 0.0
MI_QUERY = config['LSHA PARAMETERS']['MI_QUERY'] == 'True'
PLOT_DDTW = config['LSHA PARAMETERS']['PLOT_DDTW'] == 'True'
HT_QUERY = config['LSHA PARAMETERS']['HT_QUERY'] == 'True'
HT_QUERY_TYPE = config['LSHA PARAMETERS']['HT_QUERY_TYPE']
EQ_CONDITION = config['LSHA PARAMETERS']['EQ_CONDITION'].lower()


class Teacher:
    def __init__(self, sul: SystemUnderLearning, pov: str = None,
                 start_dt: str = None, end_dt: str = None, start_ts: int = None, end_ts: int = None):
        self.sul = sul

        # System-Dependent Attributes
        self.symbols = sul.symbols
        self.flows = sul.flows
        self.distributions = [v.distr for v in sul.vars]

        # Trace-Dependent Attributes
        self.timed_traces: List[TimedTrace] = sul.timed_traces
        self.signals: List[List[SampledSignal]] = sul.signals

        self.TG = TraceGenerator(pov=pov, start_dt=start_dt, end_dt=end_dt, start_ts=start_ts, end_ts=end_ts)
        self.hist = {}

    def add_distribution(self, d: ProbDistribution, f: FlowCondition):
        self.sul.add_distribution(d, f)

    # QUERIES
    @staticmethod
    def derivative(t: List[Timestamp], values: List[float]):
        # returns point-to-point increments for a given time-series
        # (derivative approximation)
        t = [x.to_secs() for x in t]
        increments = []
        try:
            increments = [(v - values[i - 1]) / (t[i] - t[i - 1]) for (i, v) in enumerate(values) if i > 0]
        except ZeroDivisionError:
            avg_dt = sum([x - t[i - 1] for (i, x) in enumerate(t) if i > 0]) / (len(t) - 1)
            increments = [(v - values[i - 1]) / avg_dt for (i, v) in enumerate(values) if i > 0]
        finally:
            return increments

    #############################################
    # MODEL IDENTIFICATION QUERY:
    # for a given prefix (word), gets all corresponding segments
    # and returns the flow condition that best fits such segments
    # If not enough data are available to draw a conclusion, returns None
    #############################################
    def mi_query(self, word: Trace):
        if not MI_QUERY or word == '':
            return self.flows[0][self.sul.default_m]
        else:
            segments = self.sul.get_segments(word)
            if len(segments) > 0:
                if len(self.flows[0]) == 1:
                    return self.flows[0][0]
                if CS == 'THERMO' and word[-1].symbol == 'h_0':
                    return self.flows[0][2]

                fits = []
                for segment in segments:
                    if len(segment) < 3:
                        continue
                    interval = [pt.timestamp for pt in segment]
                    # observed values and (approximate) derivative
                    real_behavior = [pt.value for pt in segment]
                    min_distance = 10000
                    best_fit = None

                    # for each model from the given input set
                    for flow in self.flows[0]:
                        ideal_model = flow.f(interval, segment[0].value)
                        # applies DDTW
                        res = fast_ddtw(real_behavior, ideal_model)

                        if PLOT_DDTW:
                            plot_aligned_signals(real_behavior, ideal_model, res[1])

                        if res[0] < min_distance:
                            min_distance = res[0]
                            best_fit = flow
                    else:
                        fits.append(best_fit)

                unique_fits = set(fits)
                freq = -1
                best_fit = None
                for f in unique_fits:
                    matches = sum([x == f for x in fits]) / len(fits)
                    if matches > freq:
                        freq = matches
                        best_fit = f
                if freq > 0.75:
                    return best_fit
                else:
                    LOGGER.info("!! INCONSISTENT PHYSICAL BEHAVIOR !!")
                    return None
            else:
                return None

    #############################################
    # HYPOTHESIS TESTING QUERY:
    # for a given prefix (word), gets all corresponding segments
    # and returns the random variable that best fits the randomly
    # generated model parameters.
    # If none of the available rand. variables fits the set of segments,
    # a new one is added
    # If available data are not enough to draw a conclusion, returns None
    #############################################
    def to_hist(self, values: List[float], d_id: int, update=False):
        try:
            if d_id in self.hist:
                distr = [d for d in self.distributions[0] if d.d_id == d_id][0]
                old_avg = distr.params['avg']
                old_v = len(self.hist[d_id])
                self.hist[d_id].extend(values)
            else:
                old_avg = 0.0
                old_v = 0
                self.hist[d_id] = values
            distr = [d for d in self.distributions[0] if d.d_id == d_id][0]
            distr.params['avg'] = (old_avg * old_v + sum(values)) / (old_v + len(values))
        except AttributeError:
            self.hist: Dict[int, List[float]] = {d.d_id: [] for d in self.distributions[0]}
            self.hist[d_id] = values

    def ht_query(self, word: Trace, flow: FlowCondition, save=True):
        if flow is None:
            return None

        if not HT_QUERY or word == '':
            return self.distributions[self.sul.default_d]

        if HT_QUERY_TYPE == 'D':
            return self.ht_d_query(word, flow, save)
        else:
            return self.ht_s_query(word, flow, save)

    def ht_d_query(self, word: Trace, flow: FlowCondition, save=True):
        segments = self.sul.get_segments(word)
        if len(segments) > 0:
            eligible_distributions = self.sul.vars[0].get_distr_for_flow(flow.f_id)

            metrics = [self.sul.get_ht_params(segment, flow) for segment in segments]
            metrics = [met for met in metrics if met is not None]
            unique_metrics = list(set(metrics))
            if len(unique_metrics) > 1:
                LOGGER.error('INCONSISTENT PHYSICAL BEHAVIOR')
                raise RuntimeError

            best_fit: ProbDistribution = None

            try:
                #for distr in self.hist:
                for distr in self.hist.copy():
                    value = self.hist[distr][0]
                    fits = [e_d for e_d in eligible_distributions if e_d.d_id == distr]
                    if value == unique_metrics[0] and len(fits) > 0:
                        best_fit = fits[0]
                        break
            except AttributeError:
                pass

            if best_fit is None:
                new_distr = ProbDistribution(len(self.distributions[0]), {'avg': unique_metrics[0]})
                if save:
                    self.add_distribution(new_distr, flow)
                    self.to_hist(metrics, new_distr.d_id)
                return new_distr
            else:
                self.to_hist(metrics, best_fit.d_id, update=True)
                return best_fit

    def ht_s_query(self, word: Trace, flow: FlowCondition, save=True):
        segments = self.sul.get_segments(word)
        if len(segments) > 0:
            # distr associated with selected flow
            eligible_distributions = self.sul.vars[0].get_distr_for_flow(flow.f_id)

            # randomly distributed metrics for each segment
            metrics = [self.sul.get_ht_params(segment, flow) for segment in segments]
            metrics = [met for met in metrics if met is not None]
            
            if len(metrics) == 0:
                LOGGER.warning(f"[ht_s_query] No valid metrics for word={word}, flow={flow}. Returning empty distribution.")
                return ProbDistribution(len(self.distributions[0]), {'avg': 0.0})

            avg_metrics = sum(metrics) / len(metrics)

            min_dist, best_fit = 1000, None

            try:
                #for distr in self.hist:
                for distr in self.hist.copy():
                    if len(self.hist[distr]) == 0 or len(metrics) == 0:
                        continue

                    if CS == 'THERMO':
                        v1 = metrics
                        noise1 = [0] * len(v1)
                    else:
                        v1 = [avg_metrics] * 50
                        noise1 = np.random.normal(0.0, NOISE, size=len(v1))

                    v1 = [x + noise1[i] for i, x in enumerate(v1)]

                    v2 = []
                    if CS == 'THERMO':
                        v2 = self.hist[distr]
                        noise2 = [0] * len(v2)
                    else:
                        for m in self.hist[distr]:
                            v2 += [m] * 10
                        noise2 = np.random.normal(0.0, NOISE, size=len(v2))
                    v2 = [x + noise2[i] for i, x in enumerate(v2)]

                    statistic, pvalue = stats.ks_2samp(v1, v2)
                    fits = [d for d in eligible_distributions if d.d_id == distr]
                    if statistic <= min_dist and pvalue >= P_VALUE and len(fits) > 0:
                        min_dist = statistic
                        best_fit = fits[0]
            except AttributeError:
                pass

            if best_fit is not None and min_dist < 1.0:
                self.to_hist(metrics, best_fit.d_id, update=True)
                return best_fit
            else:
                new_distr = ProbDistribution(len(self.distributions[0]), {'avg': sum(metrics) / len(metrics)})
                if save:
                    self.add_distribution(new_distr, flow)
                    self.to_hist(metrics, new_distr.d_id)
                return new_distr

    #############################################
    # ROW EQUALITY QUERY:
    # checks if two rows (row(s1), row(s2)) are weakly equal
    # returns true/false
    #############################################
    def eqr_query(self, row1: Row, row2: Row, strict=False):
        if strict:
            return row1 == row2

        for i, state in enumerate(row1.state):
            # if both rows have filled cells which differ from each other,
            # weak equality is violated
            if state.observed() and row2.state[i].observed() and state != row2.state[i]:
                return False
        else:
            return True

    #############################################
    # KNOWLEDGE REFINEMENT QUERY:
    # checks if there are ambiguous words in the observation table
    # if so, it samples new traces (through the TraceGenerator)
    # to gain more knowledge about the system under learning
    #############################################
    def ref_query(self, table: ObsTable): #parall version
        LOGGER.info('Performing ref query...')

        n_resample = int(config['LSHA PARAMETERS']['N_min'])
        S = table.get_S()
        upp_obs: List[Row] = table.get_upper_observations()
        lS = table.get_low_S()
        low_obs: List[Row] = table.get_lower_observations()

        # find all words which are ambiguous
        # (equivalent to multiple rows)
        amb_words: List[Trace] = []
        for i, row in tqdm(enumerate(upp_obs + low_obs)):
            s = S[i] if i < len(upp_obs) else lS[i - len(upp_obs)]
            for e_i, e in enumerate(table.get_E()):
                if len(self.sul.get_segments(s + e)) < n_resample:
                    amb_words.append(s + e)

            if not row.is_populated():
                continue
            
            # find equivalent rows
            eq_rows: List[Row] = []
            for (j, row_2) in enumerate(upp_obs):
                if row_2.is_populated() and i != j and self.eqr_query(row, row_2):
                    eq_rows.append(row_2)
            if len(set(eq_rows)) > 1:
                amb_words.append(s)

        # sample new traces only for ambiguous words which
        # are not prefixes of another ambiguous word
        uq = amb_words
        # for i, w in tqdm(enumerate(amb_words)):
        #     suffixes = [w2 for w2 in amb_words if w2 != w and w2.startswith(w)]
        #     if len(suffixes) == 0:
        #         uq.append(w)
       
        for word in tqdm(uq, total=len(uq)):
            LOGGER.info('Requesting new traces for {}'.format(str(word)))
            for e in table.get_E():
                self.TG.set_word(word + e)
                path = self.TG.get_traces(n_resample)
                if path is not None:
                    # Process the generated traces in parallel using process_data()
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [executor.submit(self.sul.process_data, sim) for sim in path]
                        for future in as_completed(futures):
                            try:
                                future.result()
                            except Exception as e:
                                print(f"[ref_query] Error processing sim: {e}")
                else:
                    LOGGER.debug('!! An error occurred while generating traces !!')
    #############################################
    # COUNTEREXAMPLE QUERY:
    # looks for a counterexample to current obs. table
    # returns counterexample t if:
    # -> t highlights non-closedness
    # -> t highlights non-consistency
    #############################################
    def not_closed(self, table, new_row):
        if EQ_CONDITION == 's':
            eq_rows = [row for row in table.get_upper_observations() if
                       self.eqr_query(new_row, row, strict=True)]
        else:
            eq_rows = [row for row in table.get_upper_observations() if
                       self.eqr_query(new_row, row, strict=False)]

        not_ambiguous = len(set(eq_rows)) <= 1
        diff_rows = [row for row in eq_rows if
                     not self.eqr_query(new_row, row, strict=True)]

        # FIXME: len(eq_rows)==0 does not work when rows have misaligned /bot in the signature
        # thus missing actual counterexample.
        # len(diff_rows) > 0 only works if all rows share at least one cell (as in the energy
        # CS, where all traces start with 'l').
        # A more accurate condition (TBD) should be a combination of the two.
        return len(eq_rows) == 0, not_ambiguous

    def not_consistent(self, table, S, low_S, new_row, prefix):
        for s_i, s_word in enumerate(S):
            old_row = table.get_upper_observations()[s_i] if s_i < len(S) else \
                table.get_lower_observations()[s_i - len(S)]

            # finds equal rows in S
            if EQ_CONDITION == 's':
                equal = self.eqr_query(old_row, new_row, strict=True)
            else:
                equal = self.eqr_query(old_row, new_row, strict=False)

            if equal:
                for event in self.sul.events:
                    # if the hypothetical discriminating event is already in E
                    discr_is_prefix = False
                    for e in table.get_E():
                        if str(e).startswith(event.symbol):
                            continue
                    # else checks all 1-step distant rows
                    if s_word + Trace([event]) in S:
                        old_row_a: Row = table.get_upper_observations()[
                            S.index(s_word + Trace([event]))]
                    elif s_word + Trace([event]) in low_S:
                        old_row_a: Row = table.get_lower_observations()[
                            low_S.index(s_word + Trace([event]))]
                    else:
                        continue
                    row_1_filled = old_row_a.state[0].observed()
                    row_2 = Row([])
                    for e in table.get_E():
                        id_model_2 = self.mi_query(prefix + Trace([event]) + e)
                        id_distr_2 = self.ht_query(prefix + Trace([event]) + e, id_model_2,
                                                   save=False)
                        if id_model_2 is None or id_distr_2 is None:
                            row_2.state.append(State([(None, None)]))
                        else:
                            row_2.state.append(State([(id_model_2, id_distr_2)]))
                    row_2_filled = row_2.state[0].observed()
                    if EQ_CONDITION == 's' and (row_1_filled and row_2_filled and not discr_is_prefix and
                                                not self.eqr_query(row_2, old_row_a, strict=True)):
                        return True, event, s_word
                    elif EQ_CONDITION == 'w' and (row_1_filled and row_2_filled and not discr_is_prefix and
                                                  not self.eqr_query(row_2, old_row_a, strict=False)):
                        return True, event, s_word
        return False, None, None

    def get_counterexample(self, table: ObsTable):
        LOGGER.info('Looking for counterexample...')

        S = table.get_S()
        low_S = table.get_low_S()

        # FIXME: Uncomment just for debugging purposes (i.e., to terminate the learning with a smaller table)
        # if CS == 'ENERGY' and max([len(t) for t in low_S]) >= 7:
        #    return None

        traces: List[Trace] = self.sul.traces
        not_counter: List[Trace] = []
        for i, trace in tqdm(enumerate(traces), total=len(traces)):
            for prefix in trace.get_prefixes():
                LOGGER.debug('Checking {}'.format(str(prefix)))
                # prefix is still not in table
                if prefix not in S and prefix not in low_S and prefix not in not_counter:
                    # fills hypothetical new row
                    new_row = Row([])
                    for e_i, e_word in enumerate(table.get_E()):
                        word = prefix + e_word
                        id_model = self.mi_query(word)
                        id_distr = self.ht_query(word, id_model, save=False)
                        if id_model is not None and id_distr is not None:
                            new_row.state.append(State([(id_model, id_distr)]))
                        else:
                            new_row.state.append(State([(None, None)]))
                    # if there are sufficient data to fill the new row
                    if new_row.is_populated():

                        not_closed, not_ambiguous = self.not_closed(table, new_row)
                        if not_closed:
                            # found non-closedness
                            LOGGER.warn("!! MISSED NON-CLOSEDNESS !!")
                            return prefix

                        # checks non-consistency only for rows that are not ambiguous
                        elif not_ambiguous:

                            not_consistent, event, s_word = self.not_consistent(table, S, low_S, new_row, prefix)
                            if not_consistent:
                                LOGGER.warn(
                                    "!! MISSED NON-CONSISTENCY ({}, {}) !!".format(Trace([event]), s_word))
                                return prefix
                            else:
                                not_counter.append(prefix)

                        else:
                            not_counter.append(prefix)
        else:
            # TODO: workaround to treat as a counterexample traces with events
            # that are missing from the current obs. table (which is NOT the case in L*).
            # If we want to treat as a counterexample traces that are not captured by the obs. table
            # (again, difforming from L*) including all potential combinations,
            # a more sophisticated procedure than this must be developed.
            if CS in ['ENERGY', 'AUTO_TWIN'] and len(not_counter) > 0:
                new_events = set([e.symbol for x in not_counter for e in x.events]) - \
                             set([e.symbol for t in S for e in t.events])
                if len(new_events) > 0:  # or not_counter[-1] not in S:
                    return not_counter[-1]
                else:
                    return None
            else:
                return None

    def get_counterexample_SA(self, table: ObsTable):
        LOGGER.info('Looking for counterexample (Strategy B)...')

        # 当前 observation table 中的已知前缀集合 S 和 low_S
         # S: known prefixes in the upper table; low_S: one-step-away candidates
        S = table.get_S()
        low_S = table.get_low_S()
        
        # 从 SUL（被学习系统）中获取所有 trace
        # Get all available traces from the system under learning
        traces: List[Trace] = self.sul.traces
        not_counter: List[Trace] = []

         # 一旦某线程找到 CE，就通过 found_event 通知其他线程终止
        # Once a thread finds a CE, use this Event to stop the others
        found_event = threading.Event()
        
        # 用于保护多线程下对共享结果的写入
        # Lock to safely write to shared result from multiple threads
        lock = threading.Lock()
        
        counterexample_result = [None]  

        # Function executed by each thread
        # 单个线程的任务函数：并行运行中每个线程都会执行此函数，处理分配给自己的 trace 子集
        def check_trace_group(group):
            for trace in group:
                if found_event.is_set():
                    return  # If another thread has already found a CE, stop immediately

                # 遍历该 trace 的所有前缀（prefix），逐一测试是否为 CE
                # For each prefix of this trace, test whether it is a counterexample
                for prefix in trace.get_prefixes():
                    if prefix in S or prefix in low_S or prefix in not_counter:# Skip if already seen or handled
                        continue
                    
                    # Create a new observation row for this prefix
                    new_row = Row([])
                    for e_i, e_word in enumerate(table.get_E()):
                        word = prefix + e_word
                        
                        # 模型和概率分布预测
                        # Query model ID and distribution
                        id_model = self.mi_query(word)
                        id_distr = self.ht_query(word, id_model, save=False)
                        if id_model is not None and id_distr is not None:
                            new_row.state.append(State([(id_model, id_distr)]))
                        else:
                            new_row.state.append(State([(None, None)]))

                    if new_row.is_populated():
                        not_closed, not_ambiguous = self.not_closed(table, new_row)
                        if not_closed:
                            LOGGER.warn("!! MISSED NON-CLOSEDNESS !!")
                            with lock:
                                counterexample_result[0] = prefix
                            found_event.set()
                            return

                        if not_ambiguous:
                            not_consistent, event, s_word = self.not_consistent(table, S, low_S, new_row, prefix)
                            if not_consistent:
                                LOGGER.warn("!! MISSED NON-CONSISTENCY ({}, {}) !!".format(Trace([event]), s_word))
                                with lock:
                                    counterexample_result[0] = prefix
                                found_event.set()
                                return
                            else:
                                not_counter.append(prefix)
                        else:
                            not_counter.append(prefix)

       
        n_threads = 4 
        
        # Divide the trace list into chunks for each thread
        chunk_size = (len(traces) + n_threads - 1) // n_threads
        trace_chunks = [traces[i:i + chunk_size] for i in range(0, len(traces), chunk_size)]

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(check_trace_group, chunk) for chunk in trace_chunks]
            for future in as_completed(futures):
                if found_event.is_set():
                    break  

        return counterexample_result[0]
    
    

    def get_counterexample_SB(self, table: ObsTable):
        LOGGER.info('Looking for counterexample using Strategy C (parallel + select best)...')

        S = table.get_S()
        low_S = table.get_low_S()
        traces: List[Trace] = self.sul.traces
        not_counter: List[Trace] = []

        # 分块 traces，每个线程处理一个块
        num_threads = min(8, len(traces))  # 可调线程数
        trace_chunks = [traces[i::num_threads] for i in range(num_threads)]

        def check_trace_list(trace_list: List[Trace]):
            for trace in trace_list:
                for prefix in trace.get_prefixes():
                    if prefix in S or prefix in low_S:
                        continue

                    new_row = Row([])
                    for e_word in table.get_E():
                        word = prefix + e_word
                        id_model = self.mi_query(word)
                        id_distr = self.ht_query(word, id_model, save=False)
                        new_row.state.append(State([(id_model, id_distr)]))

                    if not new_row.is_populated():
                        continue

                    not_closed, not_ambiguous = self.not_closed(table, new_row)
                    if not_closed:
                        return prefix

                    elif not_ambiguous:
                        not_consistent, event, s_word = self.not_consistent(table, S, low_S, new_row, prefix)
                        if not_consistent:
                            return prefix
            return None  # 本线程未找到 CE

        # 并行执行
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_chunk = {executor.submit(check_trace_list, chunk): chunk for chunk in trace_chunks}
            for future in as_completed(future_to_chunk):
                ce = future.result()
                if ce:
                    results.append(ce)

        if not results:
            LOGGER.info("No CE found.")
            return None
        
        def count_unique_states(trace: Trace) -> int: # Used to compute the number of unique states triggered by the trace
            seen = set()
            for e_word in table.get_E():
                word = trace + e_word
                id_model = self.mi_query(word)
                id_distr = self.ht_query(word, id_model, save=False)
                if id_model is not None and id_distr is not None:
                    seen.add((id_model, id_distr))
            return len(seen)



        def similarity_with_table(trace: Trace, table: ObsTable):# Calculate the similarity score between the trace and the most similar existing row in the observation table

            row = Row([])
            for e_word in table.get_E():
                word = trace + e_word
                id_model = self.mi_query(word)
                id_distr = self.ht_query(word, id_model, save=False)
                row.state.append(State([(id_model, id_distr)]))

            if not row.is_populated():
                return float('inf')  # 极差的相似度

            S = table.get_S()
            rows = table.get_upper_observations()
            min_similarity = float('inf')

            for s, r in zip(S, rows):
                score = sum(1 for a, b in zip(row.state, r.state) if a == b)
                min_similarity = min(min_similarity, score)

            return min_similarity
        
         # Step 3:  Weighted scoring function
        def score(trace: Trace):
            sim_score = similarity_with_table(trace)     # 越小越好
            unique_states = count_unique_states(trace)   # 越多越好
            unique_events = len(set(e.symbol for e in trace.events))  # 越多越好
            length = len(trace)                           # 越短越好

            return (
                -0.5 * sim_score +
                1.0 * unique_states +
                0.8 * unique_events +
                -0.2 * length
            )
        # Select the best counterexample based on the scoring function   
        best_ce = max(results, key=score)

        # 这个 trace 的行为与当前 ObsTable 中已有 row 最不相似，表示它能产生新信息
        # Strategy 1: Least similar to existing table rows (maximum new information)
        best_ce = min(results, key=lambda tr: similarity_with_table(tr, table))

        # 选择最优 CE（策略：最短的）
        # Strategy 2: The shortest trace
        best_ce = min(results, key=lambda tr: len(tr))
        
        # trace 涉及的唯一状态数量最多，意味着它探索了更多状态空间。
        # Strategy 3: Maximum unique states triggered
        best_ce = max(results, key=count_unique_states)
        
        # trace 包含的唯一事件 symbol 数量越多，表示更可能触发不同的系统行为。
        # Strategy 4: Maximum unique event symbols
        best_ce = max(results, key=lambda tr: len(set(e.symbol for e in tr.events)))

        
        LOGGER.success(f"Selected best CE among {len(results)} candidates: {best_ce}")
        return best_ce
