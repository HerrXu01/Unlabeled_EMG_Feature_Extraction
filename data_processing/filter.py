from scipy.signal import butter, filtfilt
from common.registry import registry

class BaseFilter:
    """
    Base class for filters, providing common functionality for applying different types of filters 
    (e.g., low-pass, band-pass) to the input data.
    """
    def __init__(self, config, data):
        self.fs = config["dataset"]["sampling_frequency"]
        self.order = config["filter"]["order"]
        self.data = data

    def apply_filter(self, btype, cutoff):
        nyquist = 0.5 * self.fs
        normalized_cutoff = [c / nyquist for c in cutoff] if isinstance(cutoff, list) else cutoff / nyquist
        b, a = butter(self.order, normalized_cutoff, btype=btype, analog=False)

        for column in self.data.columns:
            self.data[column] = filtfilt(b, a, self.data[column])

        print(f"{btype.capitalize()} filtering applied successfully.")
        return self.data


@registry.register_filter("lowpass_filter")
class LowpassFilter(BaseFilter):
    """
    Low-pass filter that removes high-frequency components from the input data, 
    using the cutoff frequency defined in the configuration.
    """
    def __init__(self, config, data):
        super().__init__(config, data)
        self.cutoff = config["filter"].get("cutoff", None)

    def apply(self):
        self.apply_filter(btype='low', cutoff=self.cutoff)


@registry.register_filter("bandpass_filter")
class BandpassFilter(BaseFilter):
    """
    Band-pass filter that retains frequencies within a specific range (between lowcut and highcut) 
    and removes frequencies outside this range.
    """
    def __init__(self, config, data):
        super().__init__(config, data)
        self.lowcut = config["filter"].get("lowcut", None)
        self.highcut = config["filter"].get("highcut", None)

    def apply(self):
        self.apply_filter(btype='band', cutoff=[self.lowcut, self.highcut])
