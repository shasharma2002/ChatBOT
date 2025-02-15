import json
import os
from datetime import datetime
from typing import Dict

class AppointmentManager:
    def __init__(self, appointments_file: str = "appointments.json"):
        self.appointments_file = appointments_file
        self.appointments = self.load_appointments()

    def load_appointments(self) -> list:
        """Load existing appointments from JSON file"""
        try:
            if os.path.exists(self.appointments_file):
                with open(self.appointments_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading appointments: {str(e)}")
            return []

    def save_appointments(self) -> None:
        """Save appointments to JSON file"""
        try:
            with open(self.appointments_file, 'w') as f:
                json.dump(self.appointments, f, indent=2)
        except Exception as e:
            print(f"Error saving appointments: {str(e)}")

    def add_appointment(self, appointment: Dict[str, str]) -> None:
        self.appointments.append(appointment)
        self.save_appointments()

    def cancel_appointment(self, idx: int) -> None:
        if 0 <= idx < len(self.appointments):
            cancelled = self.appointments.pop(idx)
            self.save_appointments()
            return f"Cancelled appointment for {cancelled['name']} on {cancelled['date']}"
        return "Appointment not found."

    def view_appointments(self) -> str:
        if not self.appointments:
            return "No appointments scheduled."
        return "\n".join([f"{idx + 1}. {appt['name']} on {appt['date']}" for idx, appt in enumerate(self.appointments)])
