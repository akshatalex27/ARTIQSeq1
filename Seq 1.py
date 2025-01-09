from artiq.experiment import *
import csv

class FullExperimentSequence12(EnvExperiment):
    def build(self):
        # Core device
        self.setattr_device("core")

        # TTL outputs
        self.setattr_device("ttl4")  # Cooling laser
        self.setattr_device("ttl5")  # Repump laser
        self.setattr_device("ttl6")  # Magnetic coils
        self.setattr_device("ttl7")  # Signal output for photon detection

        # TTL input for photon detection
        self.setattr_device("ttl0")  # Photon detection gate

        # Urukul DDS channels
        self.setattr_device("urukul0_ch0")  # Cooling laser DDS
        self.setattr_device("urukul0_ch1")  # Repump laser DDS
        self.setattr_device("urukul0_ch2")  # Atom loading DDS
        self.setattr_device("urukul0_ch3")  # Optical pumping DDS

        # Data storage
        self.time_tags = []

    def write_to_csv(self):
        """Writes the time-tagged data to a CSV file."""
        with open("photon_time_tags.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Photon Detection Event #", "Time Tag (mu)"])
            for i, time_tag in enumerate(self.time_tags):
                writer.writerow([i + 1, time_tag])
        print("Photon time-tag data has been saved to 'photon_time_tags.csv'.")


    @rpc
    def record_time_tag(self, time_tag_mu):
        """RPC to record the time-tagged photon detection data."""
        self.time_tags.append(time_tag_mu)

    @kernel
    def run(self):
        self.core.reset()

        # Main experiment parameters
        mot_load_time = 500 * ms
        atom_load_time = 100 * ms
        optical_pump_time = 10 * us
        excitation_time = 50 * ns
        detection_time = 100 * ns
        ttl7_signal_time = 100 * ns

        # Convert detection_time to machine units (for gate_rising_mu)
        detection_time_mu = self.core.seconds_to_mu(detection_time)

        max_attempts = 40

        for attempt in range(max_attempts):
            # ===== Step 1: Load MOT =====
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

            self.urukul0_ch0.sw.off()
            self.urukul0_ch1.sw.off()
            self.ttl4.off()
            self.ttl5.off()
            self.ttl6.off()

            # ===== Step 2: Load Atom =====
            self.urukul0_ch2.set(120 * MHz)
            self.urukul0_ch2.set_amplitude(0.7)
            self.urukul0_ch2.sw.on()
            delay(atom_load_time)
            self.urukul0_ch2.sw.off()

            # ===== Step 3: Optical Pumping =====
            self.urukul0_ch3.set(90 * MHz)
            self.urukul0_ch3.set_amplitude(0.6)
            self.urukul0_ch3.sw.on()
            delay(optical_pump_time)
            self.urukul0_ch3.sw.off()

            # ===== Step 4: Excitation =====
            self.urukul0_ch3.sw.on()
            delay(excitation_time)
            self.urukul0_ch3.sw.off()

            # ===== Step 5: Photon Detection =====
            photon_detected = False

            for detection_try in range(10):
                # A small slack time to move the timeline into the future.
                # Increase this if you get RTIOUnderflow.
                slack_time = 10*us
                slack_time_mu = self.core.seconds_to_mu(slack_time)

                # Get current RTIO time
                now_t = now_mu()
                # Compute a future start time for the gate
                gate_start_t = now_t + slack_time_mu

                # 1) Move the timeline to gate_start_t
                at_mu(gate_start_t)

                # 2) Open detection gate for detection_time (in MU)
                self.ttl0.gate_rising_mu(detection_time_mu)

                # The gate ends at gate_start_t + detection_time_mu
                gate_end_t = gate_start_t + detection_time_mu

                # 3) Wait for the gate duration
                delay(detection_time)

                # 4) Now read the counts up to gate_end_t
                #    Because your ARTIQ version's 'count()' requires up_to_timestamp_mu
                photon_count = self.ttl0.count(gate_end_t)

                if photon_count > 0:
                    photon_detected = True
                    current_time_tag = now_mu()
                    self.record_time_tag(current_time_tag)

                    # Indicate detection with ttl7
                    self.ttl7.on()
                    delay(ttl7_signal_time)
                    self.ttl7.off()
                    break  # If photon detected, stop this detection loop

            else:
                # If no photon is detected after 10 tries, re-cool the MOT
                self.core.break_realtime()
                self.urukul0_ch0.set(50 * MHz)
                self.urukul0_ch0.set_amplitude(0.5)
                self.urukul0_ch0.sw.on()

                self.urukul0_ch1.set(60 * MHz)
                self.urukul0_ch1.set_amplitude(0.5)
                self.urukul0_ch1.sw.on()
                self.urukul0_ch0.sw.off()
                self.urukul0_ch1.sw.off()
                continue

        # Finally, save the time-tagged data
        self.write_to_csv()
