from artiq.experiment import *
import csv
import numpy as np

class FullExperimentSequence14(EnvExperiment):  # Optimized for higher frequency
    def build(self):
        # Core device
        self.setattr_device("core")

        # TTL outputs
        self.setattr_device("ttl4")  # Cooling laser
        self.setattr_device("ttl5")  # Repump laser
        self.setattr_device("ttl6")  # Magnetic coils
        self.setattr_device("ttl7")  # Signal output (for TTL pulse)

        # TTL inputs for photon detection
        self.setattr_device("ttl0")  # Photon detection gate 1
        self.setattr_device("ttl1")  # Photon detection gate 2

        # Urukul DDS channels
        self.setattr_device("urukul0_ch0")  # Cooling laser DDS
        self.setattr_device("urukul0_ch1")  # Repump laser DDS
        self.setattr_device("urukul0_ch2")  # Atom loading DDS
        self.setattr_device("urukul0_ch3")  # Optical pumping/excitation DDS

        # Preallocate NumPy arrays for timestamps
        self.num_cycles = 100
        self.repetitions_per_cycle = 50
        self.total_reps = self.num_cycles * self.repetitions_per_cycle

        # Initialize timestamp arrays with zeros
        self.time_tags_0 = np.zeros(self.total_reps, dtype=np.int64)
        self.time_tags_1 = np.zeros(self.total_reps, dtype=np.int64)

    # RPC call to retrieve timestamp arrays
    @rpc(flags={"async"})
    def retrieve_time_tags(self):
        """Retrieve all time tags (TTL0 & TTL1) after experiment."""
        return self.time_tags_0.tolist(), self.time_tags_1.tolist()

    def write_to_csv(self, time_tags_0, time_tags_1):
        """Write all time tags (TTL0 & TTL1) to a CSV file."""
        with open("photon_time_tags.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Event #", "TTL0 Time Tag (mu)", "TTL1 Time Tag (mu)"])
            for i in range(len(time_tags_0)):
                writer.writerow([i + 1, time_tags_0[i], time_tags_1[i]])

    @kernel
    def run(self):
        self.core.reset()

        # -------------------------------
        # Experiment parameters
        # -------------------------------
        mot_load_time      = 500 * ms
        atom_load_time     = 100 * ms
        optical_pump_time  = 10 * us
        excitation_time    = 50 * ns
        pulse_width        = 50 * ns
        gate_rising_time   = 100 * ns

        # Safety margin to allow scheduler processing
        small_delay = 1 * us

        # Configure TTL directions
        self.ttl0.input()
        self.ttl1.input()
        self.ttl7.output()

        # ----------------------------------------------------------------------
        # 1) Load MOT (only once at the beginning)
        # ----------------------------------------------------------------------
        self.core.break_realtime()
        self.ttl4.on()
        self.ttl5.on()
        self.ttl6.on()

        self.urukul0_ch0.set(100 * MHz)
        self.urukul0_ch0.set_amplitude(0.8)
        self.urukul0_ch0.sw.on()

        self.urukul0_ch1.set(80 * MHz)
        self.urukul0_ch1.set_amplitude(0.5)
        self.urukul0_ch1.sw.on()

        delay(mot_load_time)

        # Turn off MOT beams and coils
        self.urukul0_ch0.sw.off()
        self.urukul0_ch1.sw.off()
        self.ttl4.off()
        self.ttl5.off()
        self.ttl6.off()

        # ----------------------------------------------------------------------
        # 2) Load Atom (once at the beginning)
        # ----------------------------------------------------------------------
        self.urukul0_ch2.set(120 * MHz)
        self.urukul0_ch2.set_amplitude(0.7)
        self.urukul0_ch2.sw.on()
        delay(atom_load_time)
        self.urukul0_ch2.sw.off()

        # ----------------------------------------------------------------------
        # 3) Configure urukul0_ch3 once before the loop
        # ----------------------------------------------------------------------
        self.urukul0_ch3.set(90 * MHz)
        self.urukul0_ch3.set_amplitude(0.6)
        self.urukul0_ch3.sw.on()

        # ----------------------------------------------------------------------
        # 4) Now loop over cycles (10 times)
        # ----------------------------------------------------------------------
        rep_index = 0  # Initialize repetition index

        for cycle_idx in range(self.num_cycles):
            for rep_idx in range(self.repetitions_per_cycle):
                # Optical Pumping using sw.pulse()
                delay(5 * us)
                self.urukul0_ch3.sw.pulse(optical_pump_time)
                delay(small_delay)  # Allow switch recovery

                # Excitation using sw.pulse()
                self.urukul0_ch3.sw.pulse(excitation_time)
                delay(small_delay)  # Allow switch recovery

                # Send TTL7 pulse & detect on ttl0 and ttl1
                with sequential:
                    # Pulse on ttl7
                    self.ttl7.pulse(pulse_width)
                    with parallel:
                        # Gate on ttl0 and ttl1
                        tend0 = self.ttl0.gate_rising(gate_rising_time)
                        tend1 = self.ttl1.gate_rising(gate_rising_time)
                # Read timestamps in machine units
                ttl_time0 = self.ttl0.timestamp_mu(tend0)
                ttl_time1 = self.ttl1.timestamp_mu(tend1)

                # Store the time tags in preallocated arrays
                self.time_tags_0[rep_index] = ttl_time0
                self.time_tags_1[rep_index] = ttl_time1

                rep_index += 1  # Increment repetition index

            # ------------------------------------------------------------------
            # 5) After repetitions_per_cycle reps, do a short "cooling" before next cycle
            # ------------------------------------------------------------------
            self.core.break_realtime()

            self.urukul0_ch0.set(50 * MHz)
            self.urukul0_ch0.set_amplitude(0.5)
            self.urukul0_ch0.sw.on()

            self.urukul0_ch1.set(60 * MHz)
            self.urukul0_ch1.set_amplitude(0.5)
            self.urukul0_ch1.sw.on()

            # Short re-cooling time
            # delay(100 * us)

            self.urukul0_ch0.sw.off()
            self.urukul0_ch1.sw.off()

        # ----------------------------------------------------------------------
        # 6) Done with all cycles: signal completion
        # ----------------------------------------------------------------------
        self.core.break_realtime()

    def analyze(self):
        """Retrieve timestamp arrays and write to CSV."""
        # Retrieve the timestamp arrays from the kernel
        time_tags_0, time_tags_1 = self.retrieve_time_tags()

        # Write the timestamps to CSV
        self.write_to_csv(time_tags_0, time_tags_1)
