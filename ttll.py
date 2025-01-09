from artiq.experiment import *
import csv

class FullExperimentSequence11(EnvExperiment):
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

        mot_load_time = 500 * ms
        atom_load_time = 100 * ms
        optical_pump_time = 10 * us
        excitation_time = 50 * ns
        detection_time = 100 * ns
        ttl7_signal_time = 100 * ns

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
                # Open detection gate for detection_time

                gate_end_mu = now_mu() + self.core.seconds_to_mu(detection_time)
                self.ttl0.gate_rising(detection_time)
                delay(detection_time)  # Wait for the detection window
                delay(10 * us)


                # Retrieve the count of rising edges up to the gate's end time
                photon_count = self.ttl0.count(gate_end_mu)


                if photon_count > 0:
                    photon_detected = True

                    # Time-tag the detection and record using @rpc
                    current_time_tag = now_mu()
                    self.record_time_tag(current_time_tag)


                    # Trigger ttl7 for 1 ms when a photon is detected
                    self.ttl7.on()
                    delay(ttl7_signal_time)
                    self.ttl7.off()
                    break  # Exit the detection loop if a photon is detected


            else:
                # Additional step if no photon is detected, back to cooling the MOT
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

        # Save the time-tagged data after the experiment

        self.write_to_csv()