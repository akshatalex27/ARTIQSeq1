from artiq.experiment import *
import numpy as np

class FullExperimentSequence14(EnvExperiment):
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

        # ------------------------------
        # Experiment hyper-parameters
        # ------------------------------
        self.num_big_cycles_chunk = 50   # Number of big cycles per chunk
        self.num_chunks           = 1    # How many such chunks to run
        self.num_cycles           = 100  # Inner cycle with Cooling
        self.repetitions_per_cycle = 50  # Repetitions for Pump/Excitation/Detection

        # The total big cycles is chunk_size * num_chunks
        # If you want e.g. 200 total big cycles, set:
        #    num_big_cycles_chunk=50, num_chunks=4, etc.

    @rpc(flags={"async"})
    def retrieve_time_tags(self):
        """Retrieve all time tags (TTL0 & TTL1) after experiment (for the chunk)."""
        return self.time_tags_0.tolist(), self.time_tags_1.tolist()

    @kernel
    def run_chunk_experiment(self, chunk_idx):
        """
        Execute one 'chunk' of the experiment consisting of self.num_big_cycles_chunk big cycles.
        All data are stored in self.time_tags_0, self.time_tags_1 arrays.
        (No type hints for 'chunk_idx', as it is not an ARTIQ type.)
        """
        self.core.reset()

        # Experiment timing constants
        mot_load_time      = 500 * ms
        atom_load_time     = 100 * ms
        optical_pump_time  = 10 * us
        excitation_time    = 50 * ns
        pulse_width        = 50 * ns
        gate_rising_time   = 100 * ns
        small_delay        = 1 * us  # Safety margin

        # Configure TTL directions
        self.ttl0.input()
        self.ttl1.input()
        self.ttl7.output()

        # Initialize repetition index for storing time tags
        rep_index = 0

        # Run the big cycles for this chunk
        for big_cycle_idx in range(self.num_big_cycles_chunk):
            # 1) Load MOT (once per big cycle)
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

            # 2) Load Atom (once per big cycle)
            self.urukul0_ch2.set(120 * MHz)
            self.urukul0_ch2.set_amplitude(0.7)
            self.urukul0_ch2.sw.on()
            delay(atom_load_time)
            self.urukul0_ch2.sw.off()

            # 3) Configure urukul0_ch3 (once per big cycle)
            self.urukul0_ch3.set(90 * MHz)
            self.urukul0_ch3.set_amplitude(0.6)
            self.urukul0_ch3.sw.on()

            # 4) Loop over inner cycles within the big cycle
            for cycle_idx in range(self.num_cycles):
                for rep in range(self.repetitions_per_cycle):
                    # Optical Pumping
                    delay(5 * us)
                    self.urukul0_ch3.sw.pulse(optical_pump_time)
                    delay(small_delay)  # switch recovery

                    # Excitation
                    self.urukul0_ch3.sw.pulse(excitation_time)
                    delay(small_delay)  # switch recovery

                    # Send TTL7 pulse & detect on ttl0 and ttl1
                    with sequential:
                        self.ttl7.pulse(pulse_width)
                        with parallel:
                            tend0 = self.ttl0.gate_rising(gate_rising_time)
                            tend1 = self.ttl1.gate_rising(gate_rising_time)

                    # Read timestamps in machine units
                    ttl_time0 = self.ttl0.timestamp_mu(tend0)
                    ttl_time1 = self.ttl1.timestamp_mu(tend1)

                    # Store the time tags
                    self.time_tags_0[rep_index] = ttl_time0
                    self.time_tags_1[rep_index] = ttl_time1
                    rep_index += 1

                # Short "cooling" or re-cooling cycle
                delay(12 * us)
                self.urukul0_ch0.set(50 * MHz)
                self.urukul0_ch0.set_amplitude(0.5)
                self.urukul0_ch0.sw.on()
                delay(12 * us)

                self.urukul0_ch1.set(60 * MHz)
                self.urukul0_ch1.set_amplitude(0.5)
                self.urukul0_ch1.sw.on()
                # Optionally a longer cooling time (e.g. delay(100 * us))

                # Turn off cooling
                self.urukul0_ch0.sw.off()
                self.urukul0_ch1.sw.off()

            # End of one big cycle
            self.core.break_realtime()

        # End of this chunk

    def run(self):
        """
        Iterate over self.num_chunks, each chunk = self.num_big_cycles_chunk big cycles.
        After each chunk, retrieve the data and store it in the dataset manager.
        """
        for chunk_idx in range(self.num_chunks):
            # 1) Pre-allocate arrays for the chunk
            self.total_reps_chunk = (self.num_big_cycles_chunk *
                                     self.num_cycles *
                                     self.repetitions_per_cycle)

            self.time_tags_0 = np.zeros(self.total_reps_chunk, dtype=np.int64)
            self.time_tags_1 = np.zeros(self.total_reps_chunk, dtype=np.int64)

            # 2) Run chunk kernel
            self.run_chunk_experiment(chunk_idx)

            # 3) Retrieve data from the kernel
            time_tags_0, time_tags_1 = self.retrieve_time_tags()

            # 4) Store the chunk data in ARTIQ’s dataset manager
            #    (You can later save or analyze them as needed)
            self.set_dataset(
                f"time_tags_0_chunk_{chunk_idx}",
                time_tags_0,
                broadcast=True
            )
            self.set_dataset(
                f"time_tags_1_chunk_{chunk_idx}",
                time_tags_1,
                broadcast=True
            )

            print(f"Chunk {chunk_idx+1} of {self.num_chunks} completed.")

        print("All chunks completed successfully.")

    def analyze(self):
        """
        Optionally perform post-processing.
        For now, data is stored chunk by chunk in the dataset manager.
        """
        pass
