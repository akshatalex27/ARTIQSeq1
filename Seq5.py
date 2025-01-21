from artiq.experiment import *
import numpy as np

class FullExperimentSequence16(EnvExperiment):
    def build(self):
        # Core device
        self.setattr_device("core")

        # TTL outputs
        self.setattr_device("ttl4")  # Cooling laser & tomography pulses
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

        # ------------------------------------------------------------------
        # Chunking parameters
        # ------------------------------------------------------------------
        # How many big cycles per chunk
        self.num_big_cycles_chunk = 80
        # How many chunks total; total big cycles = num_big_cycles_chunk * num_chunks
        self.num_chunks = 1

        # ------------------------------------------------------------------
        # Inner-sequence parameters
        # ------------------------------------------------------------------
        self.num_cooling_cycles = 100   # # of cooling cycles within each big cycle
        self.attempts_per_cooling = 50  # # of attempts in each cooling cycle

        # Each chunk has:
        #   total_attempts_chunk = (num_big_cycles_chunk * num_cooling_cycles * attempts_per_cooling)
        #
        # If (worst-case) a photon is detected in every single attempt,
        # we must allow arrays that large. Typically, many attempts won't detect a photon,
        # but we allocate a maximum capacity anyway.

        # For tomography, pick a "maximum" per-chunk (equal to total attempts is safe).
        # Alternatively, use a smaller number if you expect fewer photons.
        self.max_detected_photons_per_chunk = (
            self.num_big_cycles_chunk * self.num_cooling_cycles * self.attempts_per_cooling
        )

    # ------------------------------------------------------------------
    # RPC calls to retrieve arrays from the core device
    # ------------------------------------------------------------------
    @rpc(flags={"async"})
    def retrieve_ttl0_detections(self):
        """
        Return (array_of_detected_times_ttl0, array_of_detected_attempts_ttl0)
        as Python lists.
        """
        return (self.ttl0_detected_times.tolist(),
                self.ttl0_detected_attempts.tolist())

    @rpc(flags={"async"})
    def retrieve_ttl1_detections(self):
        """
        Return (array_of_detected_times_ttl1, array_of_detected_attempts_ttl1)
        as Python lists.
        """
        return (self.ttl1_detected_times.tolist(),
                self.ttl1_detected_attempts.tolist())

    @rpc(flags={"async"})
    def retrieve_atom_tomo_data(self):
        """
        Return (atom_tomo_times, atom_tomo_attempts) as Python lists.
        """
        return (self.atom_tomo_times.tolist(),
                self.atom_tomo_attempts.tolist())

    # ------------------------------------------------------------------
    # The kernel that runs ONE CHUNK of the experiment
    # ------------------------------------------------------------------
    @kernel
    def run_chunk_experiment(self, chunk_idx: TInt32) -> TTuple([
            TInt32,  # how many TTL0 detections
            TInt32,  # how many TTL1 detections
            TInt32   # how many tomography entries
        ]):
        """
        Execute one 'chunk' of the experiment: self.num_big_cycles_chunk big cycles.
        We do NOT store all attempts; we only store the attempts/time-tags
        for which a photon was actually detected on TTL0/TTL1.

        Return a 3-tuple:
          (final_ttl0_count, final_ttl1_count, final_tomo_count)
        so that the host code knows how many entries each array actually used.
        """
        self.core.reset()

        # -----------------------------------------------------
        # Experiment timing parameters
        # -----------------------------------------------------
        mot_load_time     = 500 * ms
        atom_load_time    = 100 * ms
        optical_pump_time = 10 * us
        excitation_time   = 50 * ns
        pulse_width       = 50 * ns
        gate_rising_time  = 100 * ns
        small_delay       = 1 * us  # safety margin

        # -----------------------------------------------------
        # Configure TTL directions
        # -----------------------------------------------------
        self.ttl0.input()
        self.ttl1.input()
        self.ttl4.output()  # used for "atom tomography" pulses
        self.ttl7.output()

        # -----------------------------------------------------
        # Local counters
        # -----------------------------------------------------
        # We'll keep track of the "attempt index" across all cycles in this chunk.
        # That attempt index increments once per "excitation/detection attempt."
        attempt_index = 0

        # Separate counters for how many detection events have been stored
        ttl0_detected_count = 0
        ttl1_detected_count = 0
        tomo_count = 0

        # -----------------------------------------------------
        # MAIN LOOP: run N = self.num_big_cycles_chunk big cycles
        # -----------------------------------------------------
        for big_cycle_idx in range(self.num_big_cycles_chunk):

            # 1) Load MOT
            self.core.break_realtime()
            self.ttl4.on()  # Turn on cooling laser
            self.ttl5.on()  # Repump on
            self.ttl6.on()  # Magnetic coils on

            self.urukul0_ch0.set(100 * MHz)
            self.urukul0_ch0.set_amplitude(0.8)
            self.urukul0_ch0.sw.on()

            self.urukul0_ch1.set(80 * MHz)
            self.urukul0_ch1.set_amplitude(0.5)
            self.urukul0_ch1.sw.on()

            delay(mot_load_time)

            # Turn off MOT beams & coils
            self.urukul0_ch0.sw.off()
            self.urukul0_ch1.sw.off()
            self.ttl4.off()
            self.ttl5.off()
            self.ttl6.off()

            # 2) Load Atom
            self.urukul0_ch2.set(120 * MHz)
            self.urukul0_ch2.set_amplitude(0.7)
            self.urukul0_ch2.sw.on()
            delay(atom_load_time)
            self.urukul0_ch2.sw.off()

            # 3) Configure urukul0_ch3 (for optical pumping + excitation)
            self.urukul0_ch3.set(90 * MHz)
            self.urukul0_ch3.set_amplitude(0.6)
            self.urukul0_ch3.sw.on()

            # 4) Up to self.num_cooling_cycles cycles
            photon_detected_this_cycle = False
            for cycle_idx in range(self.num_cooling_cycles):

                # Up to self.attempts_per_cooling attempts
                for rep in range(self.attempts_per_cooling):
                    # Optical pumping
                    delay(5 * us)
                    self.urukul0_ch3.sw.pulse(optical_pump_time)
                    delay(small_delay)

                    # Excitation
                    self.urukul0_ch3.sw.pulse(excitation_time)
                    delay(small_delay)

                    # Photon detection
                    with sequential:
                        # Send TTL pulse on ttl7
                        self.ttl7.pulse(pulse_width)
                        # Open detection gate in parallel
                        with parallel:
                            tend0 = self.ttl0.gate_rising(gate_rising_time)
                            tend1 = self.ttl1.gate_rising(gate_rising_time)

                    # Read timestamps
                    ttl_time0 = self.ttl0.timestamp_mu(tend0)
                    ttl_time1 = self.ttl1.timestamp_mu(tend1)

                    # If a photon arrives on TTL0 or TTL1 (both are possible),
                    # store them in separate arrays with attempt_index:
                    if ttl_time0 != -1:
                        self.ttl0_detected_times[ttl0_detected_count] = ttl_time0
                        self.ttl0_detected_attempts[ttl0_detected_count] = attempt_index
                        ttl0_detected_count += 1

                    if ttl_time1 != -1:
                        self.ttl1_detected_times[ttl1_detected_count] = ttl_time1
                        self.ttl1_detected_attempts[ttl1_detected_count] = attempt_index
                        ttl1_detected_count += 1

                    # If at least one was detected, do tomography and break
                    if (ttl_time0 != -1) or (ttl_time1 != -1):
                        # Record tomography time (just before tomography pulse)
                        tomography_time = now_mu()
                        self.atom_tomo_times[tomo_count] = tomography_time
                        self.atom_tomo_attempts[tomo_count] = attempt_index
                        tomo_count += 1

                        # Then do tomography (10 ms pulse on ttl4)
                        delay(3 * us)
                        self.ttl4.on()
                        delay(10 * ms)
                        self.ttl4.off()

                        photon_detected_this_cycle = True
                        break  # out of this attempt loop

                    # Increment attempt_index for every attempt
                    attempt_index += 1

                if photon_detected_this_cycle:
                    # break out of the cooling cycles
                    break

                # If no photon was found in 50 attempts, do short "cooling"
                delay(12 * us)
                self.urukul0_ch0.set(50 * MHz)
                self.urukul0_ch0.set_amplitude(0.5)
                self.urukul0_ch0.sw.on()

                delay(12 * us)
                self.urukul0_ch1.set(60 * MHz)
                self.urukul0_ch1.set_amplitude(0.5)
                self.urukul0_ch1.sw.on()

                # optionally some extra cooling
                # delay(100 * us)

                self.urukul0_ch0.sw.off()
                self.urukul0_ch1.sw.off()

            # Done with up to self.num_cooling_cycles
            self.core.break_realtime()

        # Done with self.num_big_cycles_chunk big cycles
        return (ttl0_detected_count, ttl1_detected_count, tomo_count)

    # ------------------------------------------------------------------
    # "run" orchestrates multiple chunks, storing data after each chunk
    # ------------------------------------------------------------------
    def run(self):
        """
        We run 'num_chunks' chunks. Each chunk has 'num_big_cycles_chunk' big cycles.
        After each chunk, we retrieve the data arrays from the kernel and store them.
        We only store the time tags for *actual detections* (and tomography),
        rather than all attempts.
        """
        for chunk_idx in range(self.num_chunks):
            # 1) We'll define maximum array sizes to store detection events
            # in this chunk.  In the absolute worst case, every attempt
            # could detect a photon. So we set the arrays to that length.
            chunk_max = 6*self.num_big_cycles_chunk

            # Create arrays on the host for TTL0 detections
            self.ttl0_detected_times    = np.zeros(chunk_max, dtype=np.int64)
            self.ttl0_detected_attempts = np.zeros(chunk_max, dtype=np.int64)

            # Create arrays on the host for TTL1 detections
            self.ttl1_detected_times    = np.zeros(chunk_max, dtype=np.int64)
            self.ttl1_detected_attempts = np.zeros(chunk_max, dtype=np.int64)

            # Create arrays for tomography data
            # (we also use chunk_max as the safe upper bound)
            self.atom_tomo_times    = np.zeros(chunk_max, dtype=np.int64)
            self.atom_tomo_attempts = np.zeros(chunk_max, dtype=np.int64)

            # 2) Run the kernel for this chunk
            (ttl0_count,
             ttl1_count,
             tomo_count) = self.run_chunk_experiment(chunk_idx)

            # 3) Retrieve arrays from the core device (RPC)
            #    We'll slice them on the HOST side using the returned counts.
            (ttl0_detected_times_raw,
             ttl0_detected_attempts_raw) = self.retrieve_ttl0_detections()

            (ttl1_detected_times_raw,
             ttl1_detected_attempts_raw) = self.retrieve_ttl1_detections()

            (tomo_times_raw,
             tomo_attempts_raw) = self.retrieve_atom_tomo_data()

            # 4) Slice arrays to the used length
            #    Because only the first ttl0_count slots contain valid data
            used_ttl0_times    = ttl0_detected_times_raw[:ttl0_count]
            used_ttl0_attempts = ttl0_detected_attempts_raw[:ttl0_count]

            used_ttl1_times    = ttl1_detected_times_raw[:ttl1_count]
            used_ttl1_attempts = ttl1_detected_attempts_raw[:ttl1_count]

            used_tomo_times    = tomo_times_raw[:tomo_count]
            used_tomo_attempts = tomo_attempts_raw[:tomo_count]

            # 5) Store in ARTIQ's dataset manager
            #    We'll give each chunk its own dataset name.
            self.set_dataset(
                f"ttl0_detected_times_chunk_{chunk_idx}",
                used_ttl0_times,
                broadcast=True
            )
            self.set_dataset(
                f"ttl0_detected_attempts_chunk_{chunk_idx}",
                used_ttl0_attempts,
                broadcast=True
            )

            self.set_dataset(
                f"ttl1_detected_times_chunk_{chunk_idx}",
                used_ttl1_times,
                broadcast=True
            )
            self.set_dataset(
                f"ttl1_detected_attempts_chunk_{chunk_idx}",
                used_ttl1_attempts,
                broadcast=True
            )

            self.set_dataset(
                f"atom_tomo_times_chunk_{chunk_idx}",
                used_tomo_times,
                broadcast=True
            )
            self.set_dataset(
                f"atom_tomo_attempts_chunk_{chunk_idx}",
                used_tomo_attempts,
                broadcast=True
            )

            # print(f"[CHUNK {chunk_idx+1}/{self.num_chunks}] Completed.")
            # print("  TTL0 detections:", ttl0_count)
            # print("  TTL1 detections:", ttl1_count)
            # print("  Tomography shots:", tomo_count)

    # ------------------------------------------------------------------
    # Optionally analyze or combine chunked data
    # ------------------------------------------------------------------
    def analyze(self):
        """
        Data is stored chunk by chunk in the dataset manager.
        If you want to aggregate it, you can do so here,
        or do offline post-processing after the experiment ends.
        """
        pass