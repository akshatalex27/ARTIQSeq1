from artiq.experiment import *

class FullExperimentSequence(EnvExperiment):
    def build(self):
        # Core device
        self.setattr_device("core")

        # TTL outputs
        self.setattr_device("ttl4")  # Cooling laser
        self.setattr_device("ttl5")  # Repump laser
        self.setattr_device("ttl6")  # Magnetic coils

        # TTL input for photon detection
        self.setattr_device("ttl0")  # Photon detection gate

        # Urukul DDS channels
        self.setattr_device("urukul0_ch0")  # Cooling laser DDS
        self.setattr_device("urukul0_ch1")  # Repump laser DDS
        self.setattr_device("urukul0_ch2")  # Atom loading DDS
        self.setattr_device("urukul0_ch3")  # Optical pumping DDS

    @kernel
    def run(self):
        self.core.reset()

        # Experiment parameters
        mot_load_time = 500 * ms
        atom_load_time = 100 * ms
        optical_pump_time = 10 * us  # Adjust based on requirements
        excitation_time = 50 * ns  # Adjust based on requirements
        detection_time = 100 * ns

        # DDS parameters for MOT
        cooling_freq = 100 * MHz
        cooling_ampl = 0.8
        repump_freq = 80 * MHz
        repump_ampl = 0.5

        # DDS parameters for atom loading
        atom_load_freq = 120 * MHz
        atom_load_ampl = 0.7

        # DDS parameters for optical pumping
        pump_freq = 90 * MHz
        pump_ampl = 0.6

        # Main experiment loop
        max_attempts = 5

        for attempt in range(max_attempts):
            self.log("Attempt " + str(attempt + 1) + " of " + str(max_attempts))  # Concatenate the strings for logging

            # ===== Step 1: Load MOT =====
            self.core.break_realtime()  # Ensures sufficient slack
            self.ttl4.on()  # Cooling laser
            delay(10 * us)  # Increased delay to prevent underflow
            self.ttl5.on()  # Repump laser
            delay(10 * us)
            self.ttl6.on()  # Magnetic coils
            delay(10 * us)

            # Turn on MOT DDSs
            self.urukul0_ch0.set(frequency=100 * MHz, amplitude=0.8)
            self.urukul0_ch0.sw.on()
            delay(10 * us)
            self.urukul0_ch1.set(frequency=80 * MHz, amplitude=0.5)
            self.urukul0_ch1.sw.on()
            delay(500 * ms)  # MOT loading time

            # Turn off MOT DDSs and TTLs
            self.urukul0_ch0.sw.off()
            delay(10 * us)
            self.urukul0_ch1.sw.off()
            delay(10 * us)
            self.ttl4.off()
            delay(10 * us)
            self.ttl5.off()
            delay(10 * us)
            self.ttl6.off()
            delay(10 * us)

            # ===== Step 2: Load Atom =====
            self.urukul0_ch2.set(frequency=120 * MHz, amplitude=0.7)
            self.urukul0_ch2.sw.on()
            delay(100 * ms)  # Atom loading time
            self.urukul0_ch2.sw.off()
            delay(10 * us)

            # ===== Step 3: Optical Pumping =====
            self.urukul0_ch3.set(frequency=90 * MHz, amplitude=0.6)
            self.urukul0_ch3.sw.on()
            delay(10 * us)  # Optical pumping time
            self.urukul0_ch3.sw.off()
            delay(10 * us)

            # ===== Step 4: Excitation =====
            self.urukul0_ch3.sw.on()  # Use same DDS for excitation
            delay(50 * ns)  # Excitation pulse
            self.urukul0_ch3.sw.off()
            delay(10 * us)

            # ===== Step 5: Photon Detection (Attempt 1) =====
            photon_detected = False
            for detection_try in range(10):
                self.log("Photon detection attempt " + str(detection_try + 1) + " of 10")  # Concatenate the strings for logging
                self.ttl0.gate_rising(100 * ns)
                delay(10 * us)  # Delay before reading counts
                if self.ttl0.count(now_mu()) > 0:
                    # Photon detected
                    self.log("Photon detected!")  # Use log instead of print
                    photon_detected = True
                    break
                else:
                    self.log("No photon detected.")  # Use log instead of print

            if photon_detected:
                # Photon detected: Return to MOT loading
                self.log("Returning to MOT loading after photon detection.")  # Use log instead of print
                delay(10 * us)  # Delay before restarting loop
                continue
            else:
                # No photon detected after 10 tries: Perform cooling and restart from MOT and Atom loading
                self.log("No photon detected after 10 tries. Performing cooling.")  # Use log instead of print
                self.urukul0_ch0.set(frequency=50 * MHz, amplitude=0.5)  # Cooling DDS
                self.urukul0_ch0.sw.on()
                self.urukul0_ch1.set(frequency=60 * MHz, amplitude=0.5)  # Cooling DDS
                self.urukul0_ch1.sw.on()
                delay(100 * us)  # Cooling time (100 microseconds)
                self.urukul0_ch0.sw.off()
                self.urukul0_ch1.sw.off()
                delay(10 * us)

                # Start from MOT and Atom Loading
                continue  # Restart the whole experiment

        # If no photon was detected in 5 attempts, start from MOT loading again
        self.log("No photon detected after 5 full attempts. Restarting from MOT loading.")  # Use log instead of print
        self.core.reset()  # Reset the experiment