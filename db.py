"""
db.py  ──────────────────────────────────────────────────────────
Central database + ORM layer for the Hotel-Labor-Tool project.
SQLite for development; swap the connection-string in ENGINE
for Postgres/MySQL when you’re ready for production.
"""

from sqlalchemy import (
      create_engine, Column, Integer, Float, String,
      Date, ForeignKey, UniqueConstraint, func, text, Time
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship

# ───────────────────────────────────────────────────────────────
#  ENGINE  &  SESSION
# ───────────────────────────────────────────────────────────────
ENGINE = create_engine("sqlite:///hotel_labor.db", echo=False)
Session = sessionmaker(bind=ENGINE)
Base = declarative_base()

# ───────────────────────────────────────────────────────────────
#  MASTER DATA
# ───────────────────────────────────────────────────────────────
class Department(Base):
      __tablename__ = "departments"

      id   = Column(Integer, primary_key=True)
      name = Column(String, unique=True, nullable=False)         # e.g. Rooms, F&B


class Position(Base):
      __tablename__ = "positions"

      id            = Column(Integer, primary_key=True)
      name          = Column(String,  nullable=False)            # e.g. Room Attendant
      department_id = Column(Integer, ForeignKey("departments.id"))
      __table_args__ = (UniqueConstraint("name", "department_id",
                                         name="uix_pos_dept"),)
class Schedule(Base):
    __tablename__ = "schedule"

    id         = Column(Integer, primary_key=True)
    emp_id     = Column(Integer, ForeignKey("employee.id"), nullable=False)
    day        = Column(Date, nullable=False)             # ← change from String to Date
    shift_type = Column(String(20), nullable=False)

    __table_args__ = (UniqueConstraint("emp_id", "day", name="uix_emp_day"),)

# In db.py
class PositionShift(Base):
    __tablename__ = "position_shift"

    id = Column(Integer, primary_key=True)
    department = Column(String)
    position = Column(String)
    shift_period = Column(String)  # "Morning", "Afternoon", "Evening"
    shift_time = Column(String)    # e.g., "9:00 AM - 5:30 PM"

class ShiftTime(Base):
    __tablename__ = "shift_times"
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey("positions.id"))
    period = Column(String)  # Morning, Afternoon, Evening
    start = Column(Time)
    end = Column(Time)

class RoomActual(Base):
    __tablename__ = "room_actual"
    id = Column(Integer, primary_key=True)
    kpi = Column(String)
    date = Column(Date)
    value = Column(Integer)

class RoomForecast(Base):
    __tablename__ = "room_forecast"
    id = Column(Integer, primary_key=True)
    kpi = Column(String)
    date = Column(Date)
    value = Column(Integer)

class ScheduleAvailability(Base):
    __tablename__ = "schedule_availability"
    id = Column(Integer, primary_key=True)
    emp_id = Column(Integer, ForeignKey("employee.id"), nullable=False)
    weekday = Column(String, nullable=False)      # "Monday", "Tuesday", etc.
    availability = Column(String, nullable=False) # "Morning", "Afternoon", or "OFF"

    employee = relationship("Employee", backref="availabilities")

class LaborStandard(Base):
    __tablename__ = "labor_standards"
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey("positions.id"))
    metric = Column(String)
    standard = Column(Float)
    unit = Column(String, default="per FTE")

class RoomOTBPickup(Base):
    __tablename__ = "room_otb_pickup"

    id     = Column(Integer, primary_key=True)
    date   = Column(Date, nullable=False)
    kpi    = Column(String, nullable=False)
    value  = Column(Integer, default=0)

class OTBShift(Base):
      __tablename__ = "otb_shift"
      id = Column(Integer, primary_key=True)
      position_id = Column(Integer)  # ← no ForeignKey to avoid dependency issues
      date = Column(Date)
      hours = Column(Float)

from sqlalchemy import Column, String, Date, Float, Integer

class PlanningSummary(Base):
    __tablename__ = "planning_summary"
    id = Column(Integer, primary_key=True)
    position = Column(String)
    date = Column(Date)
    scheduled_hours = Column(Float)
    fte = Column(Float)

class ProjectedHours(Base):
    __tablename__ = "projected_hours"
    id = Column(Integer, primary_key=True)
    position = Column(String)
    date = Column(Date)
    otb_hours = Column(Float)
    fte = Column(Float)

class OTBHours(Base):
    __tablename__ = "otb_hours"
    id = Column(Integer, primary_key=True)
    position = Column(String)
    date = Column(Date)
    otb_hours = Column(Float)
    fte = Column(Float)

# ───────────────────────────────────────────────────────────────
#  EMPLOYEES  (single definition)
# ───────────────────────────────────────────────────────────────
class Employee(Base):
      """
      Keeps hourly_rate so we can derive Labor-$ metrics quickly.
      emp_type:  "import"  = In-House
                 "manual"  = Contract Labor
      """
      __tablename__ = "employee"
      __table_args__ = {'extend_existing': True}

      id          = Column(Integer, primary_key=True)
      name        = Column(String, nullable=False)               # “Perez, Jesus 123”
      role        = Column(String, nullable=False)
      department  = Column(String, nullable=False)
      hourly_rate = Column(Float,  nullable=False)
      emp_type    = Column(String(15), nullable=False, default="import")

# ───────────────────────────────────────────────────────────────
#  ACTUAL HOURS
# ───────────────────────────────────────────────────────────────
class Actual(Base):
      __tablename__ = "actual"

      id          = Column(Integer, primary_key=True)
      emp_id      = Column(Integer, ForeignKey("employee.id"))
      position_id = Column(Integer, ForeignKey("positions.id"))
      date        = Column(Date, nullable=False)

      hours    = Column(Float, default=0.0)
      ot_hours = Column(Float, default=0.0)
      reg_pay  = Column(Float, default=0.0)
      ot_pay   = Column(Float, default=0.0)
      source   = Column(String, default="manual")  # "manual" | "contract"

# ───────────────────────────────────────────────────────────────
#  ROOMS
# ───────────────────────────────────────────────────────────────
class Rooms(Base):
      __tablename__ = "rooms"

      id       = Column(Integer, primary_key=True)
      date     = Column(Date, nullable=False, unique=True)
      occupied = Column(Integer, nullable=False)

# ───────────────────────────────────────────────────────────────
#  INITIALISER
# ───────────────────────────────────────────────────────────────
def init_db() -> None:
      """Creates all tables if they don’t exist."""
      Base.metadata.create_all(ENGINE)

# ── ONE-TIME patch: add emp_type column if it’s missing ────────
with ENGINE.connect() as con:
      try:
            con.execute(text(
                  'ALTER TABLE employee ADD COLUMN emp_type TEXT NOT NULL DEFAULT "import";'
            ))
      except Exception as e:
            # duplicate column -> safe to ignore on subsequent runs
            print("Skipping emp_type column creation:", e)

# Create tables (if first run) after the patch
init_db()

if __name__ == "__main__":
      init_db()
      print("✅ hotel_labor.db initialised / upgraded.")
