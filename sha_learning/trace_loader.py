import os

def load_and_process_all_traces(trace_dir: str, teacher) -> list:
   
    trace_files = sorted(f for f in os.listdir(trace_dir) if f.startswith("trace_") and f.endswith(".txt"))

    traces = []
    for filename in trace_files:
        path = os.path.join(trace_dir, filename)
        trace = teacher.sul.process_data(path)  
        if trace is not None:  
            traces.append(trace)
        else:
            print(f"can't analyse trace : {filename}")

    return traces