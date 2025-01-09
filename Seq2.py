from artiq.experiment import *
import csv

class FullExperimentSequence13(EnvExperiment):#currently working with 2.5Khz frequency
    def build(self):
        # Core device
        self.setattr_device("core")

        # TTL outputs
        self.setattr_device("ttl4")  # Cooling laser
        self.setattr_device("ttl5")  # Repump laser
        self.setattr_device("ttl6")  # Magnetic coils
        self.setattr_device("ttl7")  # Signal output (for TTL pulse)

        # TTL inputs for photon detection
        self.setattr_device("ttl0")
        self.setattr_device("ttl1")

        # Urukul DDS channels
        self.setattr_device("urukul0_ch0")  # Cooling laser DDS
        self.setattr_device("urukul0_ch1")  # Repump laser DDS
        self.setattr_device("urukul0_ch2")  # Atom loading DDS
        self.setattr_device("urukul0_ch3")  # Optical pumping/excitation DDS

        # Two lists for storing timestamps from ttl0 & ttl1
        self.time_tags_0 = []
        self.time_tags_1 = []

    # RPC calls to store time tags
    @rpc
    def record_time_tag_0(self, t_mu: TInt64):
        self.time_tags_0.append(t_mu)

    @rpc
    def record_time_tag_1(self, t_mu: TInt64):
        self.time_tags_1.append(t_mu)

    def write_to_csv(self):
        """Write all time tags (channel 0 & channel 1) to CSV."""
        with open("photon_time_tags.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Event #", "TTL0 Time Tag (mu)", "TTL1 Time Tag (mu)"])
            # We'll assume both lists have the same length: one entry per repetition
            for i in range(len(self.time_tags_0)):
                writer.writerow([i+1, self.time_tags_0[i], self.time_tags_1[i]])

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

        # Repetitions
        repetitions_per_cycle = 50
        num_cycles            = 10

        # TTL direction configuration
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

        # Turn off MOT beams + coils
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
        # 3) Now do 10 cycles, each with 50 reps
        # ----------------------------------------------------------------------
        for cycle_idx in range(num_cycles):
            for rep_idx in range(repetitions_per_cycle):

                # --------------------------------------------------------------
                # a) Optical Pumping
                # --------------------------------------------------------------
                self.core.break_realtime()
                self.urukul0_ch3.set(90 * MHz)
                self.urukul0_ch3.set_amplitude(0.6)
                self.urukul0_ch3.sw.on()
                delay(optical_pump_time)
                self.urukul0_ch3.sw.off()

                # --------------------------------------------------------------
                # b) Excitation
                # --------------------------------------------------------------
                self.urukul0_ch3.sw.on()
                delay(excitation_time)
                self.urukul0_ch3.sw.off()

                # --------------------------------------------------------------
                # c) Send TTL7 pulse & detect on ttl0, ttl1
                #    (Snippet from your “2nd code,” minus prints)
                # --------------------------------------------------------------
                # self.core.break_realtime()
                # delay(1*us)  # small margin so no underflow

                with sequential:
                    # Pulse on ttl7
                    self.ttl7.pulse(pulse_width)
                    with parallel:

                        # Gate on ttl0 and ttl1
                        tend0 = self.ttl0.gate_rising(100 * ns)
                        tend1 = self.ttl1.gate_rising(100 * ns)



                # Read timestamps in machine units
                ttl_time0 = self.ttl0.timestamp_mu(tend0)
                ttl_time1 = self.ttl1.timestamp_mu(tend1)

                # --------------------------------------------------------------
                # d) Store the time tags in Python (no printing)
                # --------------------------------------------------------------
                self.record_time_tag_0(ttl_time0)
                self.record_time_tag_1(ttl_time1)

            # ------------------------------------------------------------------
            # e) After 50 reps, do a short "cooling" before next cycle
            # ------------------------------------------------------------------
            self.core.break_realtime()

            self.urukul0_ch0.set(50 * MHz)
            self.urukul0_ch0.set_amplitude(0.5)
            self.urukul0_ch0.sw.on()

            self.urukul0_ch1.set(60 * MHz)
            self.urukul0_ch1.set_amplitude(0.5)
            self.urukul0_ch1.sw.on()

            # delay(100 * us)
            self.urukul0_ch0.sw.off()
            self.urukul0_ch1.sw.off()

        # ----------------------------------------------------------------------
        # 4) Done with all cycles: save time-tag data to CSV
        # ----------------------------------------------------------------------
        self.core.break_realtime()
        self.write_to_csv()
