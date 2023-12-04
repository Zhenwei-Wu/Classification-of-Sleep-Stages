def read_edf(file_path):
    with pyedflib.EdfReader(file_path) as edf:
        signals = [edf.readSignal(i) for i in range(edf.signals_in_file)]
        signal_labels = edf.getSignalLabels()
        signal_headers = edf.getSignalHeaders()
    return signals, signal_labels, signal_headers

# Import the PSG and Hypnogram files
psg_signals, psg_labels, psg_headers = read_edf(psg_file_path)

def read_hypnogram(file_path):
    with pyedflib.EdfReader(file_path) as edf:
        annotations = edf.readAnnotations()
    return annotations