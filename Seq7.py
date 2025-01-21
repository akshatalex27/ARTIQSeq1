from artiq.experiment import *
import numpy as np
import time

class FullExperimentSequence21(EnvExperiment):
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

        # ------------------------------------------------------------------
        # Chunking parameters
        # ------------------------------------------------------------------
        self.num_big_cycles_chunk = 80
        self.num_chunks = 1

        # ------------------------------------------------------------------
        # Inner-sequence parameters
        # ------------------------------------------------------------------
        self.num_cooling_cycles = 100
        self.attempts_per_cooling = 50

        # ------------------------------------------------------------------
        # Arrays needed for time-tagging photon detections
        # (We allocate enough space to store detection events.)
        # ------------------------------------------------------------------
        # You can adjust this to ensure you have enough space.
        self.max_detected_photons_per_chunk = (
            self.num_big_cycles_chunk * self.num_cooling_cycles * self.attempts_per_cooling
        )

    # ------------------------------------------------------------------
    # RPC calls to retrieve arrays from the core device (TTL0/TTL1)
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

    # ------------------------------------------------------------------
    # RPC to wait on the host side (outside the timeline)
    # ------------------------------------------------------------------
    @rpc
    def host_mot_load_wait(self, seconds):
        """
        Wait on the host side for `seconds` (using time.sleep).
        """
        time.sleep(seconds)

    # ------------------------------------------------------------------
    # The kernel that runs ONE CHUNK of the experiment
    # ------------------------------------------------------------------
    @kernel
    def run_chunk_experiment(self, chunk_idx: TInt32) -> TTuple([
            TInt32,  # how many TTL0 detections
            TInt32   # how many TTL1 detections
        ]):
        """
        Execute one 'chunk' (self.num_big_cycles_chunk big cycles).
        We store attempt/time-tags for any detected photons (TTL0/TTL1).
        Unlike the original version, we do NOT break out of the loops
        when a photon is detected. We simply complete all 50 attempts
        per cooling cycle, and all 100 cooling cycles.
        Return (ttl0_count, ttl1_count).
        """
        self.core.reset()

        # -----------------------------------------------------
        # Experiment timing parameters
        # -----------------------------------------------------
        mot_load_time     = 500 * ms  # SHIFTED to host-side wait
        atom_load_time    = 100 * ms
        optical_pump_time = 10 * us
        excitation_time   = 50 * ns
        pulse_width       = 50 * ns
        gate_rising_time  = 100 * ns
        small_delay       = 1 * us  # small safety margin

        # -----------------------------------------------------
        # Configure TTL directions
        # -----------------------------------------------------
        self.ttl0.input()
        self.ttl1.input()
        self.ttl4.output()
        self.ttl7.output()

        # Local counters
        attempt_index = 0
        ttl0_detected_count = 0
        ttl1_detected_count = 0

        # -----------------------------------------------------
        # MAIN LOOP: run N = self.num_big_cycles_chunk big cycles
        # -----------------------------------------------------
        for big_cycle_idx in range(self.num_big_cycles_chunk):

            # 1) Turn on MOT beams & coils
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

            # 2) Wait 500 ms on the host side (not blocking RTIO timeline)
            self.core.break_realtime()
            self.host_mot_load_wait(0.5)

            # Re-sync timeline after host wait
            self.core.break_realtime()

            # 3) Turn off MOT beams & coils
            self.urukul0_ch0.sw.off()
            self.urukul0_ch1.sw.off()
            self.ttl4.off()
            self.ttl5.off()
            self.ttl6.off()

            # 4) Load Atom (100 ms in RTIO timeline)
            self.urukul0_ch2.set(120 * MHz)
            self.urukul0_ch2.set_amplitude(0.7)
            self.urukul0_ch2.sw.on()
            delay(atom_load_time)
            self.urukul0_ch2.sw.off()

            # 5) Configure urukul0_ch3 (for optical pumping + excitation)
            self.urukul0_ch3.set(90 * MHz)
            self.urukul0_ch3.set_amplitude(0.6)
            self.urukul0_ch3.sw.on()

            # 6) Start cooling cycles (each with multiple attempts)
            for cycle_idx in range(self.num_cooling_cycles):

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
                        self.ttl7.pulse(pulse_width)  # TTL pulse out for reference
                        with parallel:
                            tend0 = self.ttl0.gate_rising(gate_rising_time)
                            tend1 = self.ttl1.gate_rising(gate_rising_time)

                    # Check timestamps and store if present
                    ttl_time0 = self.ttl0.timestamp_mu(tend0)
                    ttl_time1 = self.ttl1.timestamp_mu(tend1)

                    if ttl_time0 != -1:
                        self.ttl0_detected_times[ttl0_detected_count] = ttl_time0
                        self.ttl0_detected_attempts[ttl0_detected_count] = attempt_index
                        ttl0_detected_count += 1

                    if ttl_time1 != -1:
                        self.ttl1_detected_times[ttl1_detected_count] = ttl_time1
                        self.ttl1_detected_attempts[ttl1_detected_count] = attempt_index
                        ttl1_detected_count += 1

                    # Increment attempt index after every attempt
                    attempt_index += 1

                # After finishing all attempts in this cooling cycle, do a short cooling.
                delay(12 * us)
                self.urukul0_ch0.set(50 * MHz)
                self.urukul0_ch0.set_amplitude(0.5)
                self.urukul0_ch0.sw.on()

                delay(12 * us)
                self.urukul0_ch1.set(60 * MHz)
                self.urukul0_ch1.set_amplitude(0.5)
                self.urukul0_ch1.sw.on()

                # optionally extra cooling
                # delay(100 * us)

                self.urukul0_ch0.sw.off()
                self.urukul0_ch1.sw.off()

            # End of num_cooling_cycles
            self.core.break_realtime()

        # End of self.num_big_cycles_chunk big cycles
        return (ttl0_detected_count, ttl1_detected_count)

    # ------------------------------------------------------------------
    # "run" orchestrates multiple chunks, storing data after each chunk
    # ------------------------------------------------------------------
    def run(self):
        """
        We run 'num_chunks' chunks. Each chunk has 'num_big_cycles_chunk' big cycles.
        After each chunk, we retrieve arrays and store them.
        """
        for chunk_idx in range(self.num_chunks):
            # We allocate arrays on the host side for storing detection events
            # Potential maximum needed:
            chunk_max = (
                self.num_big_cycles_chunk * self.num_cooling_cycles * 10
            )

            # Create arrays on the host
            self.ttl0_detected_times = np.zeros(chunk_max, dtype=np.int64)
            self.ttl0_detected_attempts = np.zeros(chunk_max, dtype=np.int64)
            self.ttl1_detected_times = np.zeros(chunk_max, dtype=np.int64)
            self.ttl1_detected_attempts = np.zeros(chunk_max, dtype=np.int64)

            # Run the kernel for this chunk
            (ttl0_count,
             ttl1_count) = self.run_chunk_experiment(chunk_idx)

            # Retrieve arrays (host side)
            (ttl0_detected_times_raw,
             ttl0_detected_attempts_raw) = self.retrieve_ttl0_detections()
            (ttl1_detected_times_raw,
             ttl1_detected_attempts_raw) = self.retrieve_ttl1_detections()

            # Slice arrays to the valid used length
            used_ttl0_times = ttl0_detected_times_raw[:ttl0_count]
            used_ttl0_attempts = ttl0_detected_attempts_raw[:ttl0_count]
            used_ttl1_times = ttl1_detected_times_raw[:ttl1_count]
            used_ttl1_attempts = ttl1_detected_attempts_raw[:ttl1_count]

            # Store in ARTIQ dataset manager
            self.set_dataset(f"ttl0_detected_times_chunk_{chunk_idx}",
                             used_ttl0_times, broadcast=True)
            self.set_dataset(f"ttl0_detected_attempts_chunk_{chunk_idx}",
                             used_ttl0_attempts, broadcast=True)

            self.set_dataset(f"ttl1_detected_times_chunk_{chunk_idx}",
                             used_ttl1_times, broadcast=True)
            self.set_dataset(f"ttl1_detected_attempts_chunk_{chunk_idx}",
                             used_ttl1_attempts, broadcast=True)

    # ------------------------------------------------------------------
    # analyze (optional)
    # ------------------------------------------------------------------
    def analyze(self):
        """
        Perform any post-processing if desired.
        """
        pass
