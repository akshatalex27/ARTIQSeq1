from artiq.experiment import *
import numpy as np
import time

# We add these two imports for handling the HDF5 file.
import h5py
import datetime

class FullExperimentSequence17(EnvExperiment):   # atom-photon entanglement sequence
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
        self.num_big_cycles_chunk = 20
        self.num_chunks = 3  # example: do 3 chunks

        # ------------------------------------------------------------------
        # Inner-sequence parameters
        # ------------------------------------------------------------------
        self.num_cooling_cycles = 100
        self.attempts_per_cooling = 50

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
    # RPC to wait on the host side (outside the timeline)
    # ------------------------------------------------------------------
    @rpc
    def host_mot_load_wait(self, seconds):
        """
        Wait on the host side for seconds (using time.sleep).
        """
        time.sleep(seconds)

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
        Execute one 'chunk' (self.num_big_cycles_chunk big cycles).
        We only store attempt/time-tags for which a photon was detected.
        Return (ttl0_count, ttl1_count, tomo_count).
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
        small_delay       = 1 * us  # safety margin

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
        tomo_count = 0

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

            # 2) Wait 500 ms on the host side (not blocking the RTIO timeline)
            self.core.break_realtime()
            self.host_mot_load_wait(0.5)

            # IMPORTANT: Re-sync timeline after host wait
            self.core.break_realtime()

            # 3) Turn off MOT beams & coils
            self.urukul0_ch0.sw.off()
            self.urukul0_ch1.sw.off()
            self.ttl4.off()
            self.ttl5.off()
            self.ttl6.off()

            # 4) Load Atom (100 ms in RTIO timeline is okay)
            self.urukul0_ch2.set(120 * MHz)
            self.urukul0_ch2.set_amplitude(0.7)
            self.urukul0_ch2.sw.on()
            delay(atom_load_time)
            self.urukul0_ch2.sw.off()

            # 5) Configure urukul0_ch3 (for optical pumping + excitation)
            self.urukul0_ch3.set(90 * MHz)
            self.urukul0_ch3.set_amplitude(0.6)
            self.urukul0_ch3.sw.on()

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
                        self.ttl7.pulse(pulse_width)  # TTL pulse out
                        with parallel:
                            tend0 = self.ttl0.gate_rising(gate_rising_time)
                            tend1 = self.ttl1.gate_rising(gate_rising_time)

                    # Check timestamps
                    ttl_time0 = self.ttl0.timestamp_mu(tend0)
                    ttl_time1 = self.ttl1.timestamp_mu(tend1)

                    # Store TTL0 detection if present
                    if ttl_time0 != -1:
                        self.ttl0_detected_times[ttl0_detected_count] = ttl_time0
                        self.ttl0_detected_attempts[ttl0_detected_count] = attempt_index
                        ttl0_detected_count += 1

                    # Store TTL1 detection if present
                    if ttl_time1 != -1:
                        self.ttl1_detected_times[ttl1_detected_count] = ttl_time1
                        self.ttl1_detected_attempts[ttl1_detected_count] = attempt_index
                        ttl1_detected_count += 1

                    # If at least one photon was detected, do tomography & break
                    if (ttl_time0 != -1) or (ttl_time1 != -1):
                        tomography_time = now_mu()
                        self.atom_tomo_times[tomo_count] = tomography_time
                        self.atom_tomo_attempts[tomo_count] = attempt_index
                        tomo_count += 1

                        # 10 ms tomography pulse on ttl4
                        delay(3 * us)
                        self.ttl4.on()
                        delay(10 * ms)
                        self.ttl4.off()

                        photon_detected_this_cycle = True
                        break  # out of attempts loop

                    # Increment attempt index after every attempt
                    attempt_index += 1

                # If photon was detected this cycle, move on to the next big cycle
                if photon_detected_this_cycle:
                    break

                # If no photon in 50 attempts, do short "cooling"
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

            # End of self.num_cooling_cycles
            self.core.break_realtime()

        # End of self.num_big_cycles_chunk big cycles
        return (ttl0_detected_count, ttl1_detected_count, tomo_count)

    # ------------------------------------------------------------------
    # "run" orchestrates multiple chunks, storing data after each chunk
    # ------------------------------------------------------------------
    def run(self):
        """
        We run 'num_chunks' chunks. Each chunk has 'num_big_cycles_chunk' big cycles.
        After each chunk, we retrieve arrays, store them in the dataset manager
        (for live display), and also store them in an HDF5 file, flushing so that
        partial data is preserved if the experiment is stopped.
        """
        # Create a unique filename for our HDF5 file
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"atom_photon_data_{timestamp_str}.h5"

        # Open the HDF5 file in write mode
        self.h5file = h5py.File(filename, "w")

        # We put our main logic in a try/finally so that
        # if the experiment is terminated, we still close the file gracefully.
        try:
            for chunk_idx in range(self.num_chunks):
                # For each chunk, define arrays on the host side that we will use in the kernel.
                # (We use an over-estimated size just to ensure no out-of-bounds in kernel.)
                chunk_max = 6 * self.num_big_cycles_chunk
                self.ttl0_detected_times = np.zeros(chunk_max, dtype=np.int64)
                self.ttl0_detected_attempts = np.zeros(chunk_max, dtype=np.int64)
                self.ttl1_detected_times = np.zeros(chunk_max, dtype=np.int64)
                self.ttl1_detected_attempts = np.zeros(chunk_max, dtype=np.int64)
                self.atom_tomo_times = np.zeros(chunk_max, dtype=np.int64)
                self.atom_tomo_attempts = np.zeros(chunk_max, dtype=np.int64)

                # Run the kernel for this chunk
                (ttl0_count,
                 ttl1_count,
                 tomo_count) = self.run_chunk_experiment(chunk_idx)

                # Retrieve arrays (host side)
                (ttl0_detected_times_raw,
                 ttl0_detected_attempts_raw) = self.retrieve_ttl0_detections()
                (ttl1_detected_times_raw,
                 ttl1_detected_attempts_raw) = self.retrieve_ttl1_detections()
                (tomo_times_raw,
                 tomo_attempts_raw) = self.retrieve_atom_tomo_data()

                # Slice arrays to valid used length
                used_ttl0_times = ttl0_detected_times_raw[:ttl0_count]
                used_ttl0_attempts = ttl0_detected_attempts_raw[:ttl0_count]
                used_ttl1_times = ttl1_detected_times_raw[:ttl1_count]
                used_ttl1_attempts = ttl1_detected_attempts_raw[:ttl1_count]
                used_tomo_times = tomo_times_raw[:tomo_count]
                used_tomo_attempts = tomo_attempts_raw[:tomo_count]

                # -----------------------------------------------------------
                # 1) Store to ARTIQ dataset manager (for real-time plotting)
                # -----------------------------------------------------------
                self.set_dataset(f"ttl0_detected_times_chunk_{chunk_idx}",
                                 used_ttl0_times, broadcast=True)
                self.set_dataset(f"ttl0_detected_attempts_chunk_{chunk_idx}",
                                 used_ttl0_attempts, broadcast=True)
                self.set_dataset(f"ttl1_detected_times_chunk_{chunk_idx}",
                                 used_ttl1_times, broadcast=True)
                self.set_dataset(f"ttl1_detected_attempts_chunk_{chunk_idx}",
                                 used_ttl1_attempts, broadcast=True)
                self.set_dataset(f"atom_tomo_times_chunk_{chunk_idx}",
                                 used_tomo_times, broadcast=True)
                self.set_dataset(f"atom_tomo_attempts_chunk_{chunk_idx}",
                                 used_tomo_attempts, broadcast=True)

                # -----------------------------------------------------------
                # 2) Store to our *own* HDF5 file
                # -----------------------------------------------------------
                g = self.h5file.create_group(f"chunk_{chunk_idx}")
                g.create_dataset("ttl0_times", data=used_ttl0_times)
                g.create_dataset("ttl0_attempts", data=used_ttl0_attempts)
                g.create_dataset("ttl1_times", data=used_ttl1_times)
                g.create_dataset("ttl1_attempts", data=used_ttl1_attempts)
                g.create_dataset("tomo_times", data=used_tomo_times)
                g.create_dataset("tomo_attempts", data=used_tomo_attempts)

                # Flush to disk so if we terminate now, chunk data is not lost.
                self.h5file.flush()

        finally:
            # If the experiment is forcibly terminated (Dashboard -> cancel),
            # ARTIQ raises a termination exception. The `finally` block ensures
            # we close the file gracefully, leaving it in a valid state.
            self.h5file.close()
            # print(f"*** HDF5 file '{filename}' closed. Partial data is safe. ***")

    # ------------------------------------------------------------------
    # Optionally analyze or combine chunked data
    # ------------------------------------------------------------------
    def analyze(self):
        """
        Perform any post-processing here if desired.
        """
        pass
