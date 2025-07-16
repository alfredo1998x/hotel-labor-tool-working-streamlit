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
from sqlalchemy.orm import sessionmaker, relationship

# ────────────── ENGINE / SESSION ──────────────
ENGINE = create_engine("sqlite:///hotel_labor.db", echo=False)
Session = sessionmaker(bind=ENGINE)
Base = declarative_base()

# ────────────── HOTEL SCOPING ──────────────
class HotelScoped:
    hotel_name = Column(String, nullable=False)

# ────────────── MODELS ──────────────
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    hotel_name = Column(String, nullable=False)
    role = Column(String, default="manager")

class Department(Base, HotelScoped):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

class Position(Base, HotelScoped):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id"))
    __table_args__ = (UniqueConstraint("name", "department_id", name="uix_pos_dept"),)

class Employee(Base, HotelScoped):
    __tablename__ = "employee"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    department = Column(String, nullable=False)
    hourly_rate = Column(Float, nullable=False)
    emp_type = Column(String(15), nullable=False, default="import")

class Schedule(Base, HotelScoped):
    __tablename__ = "schedule"
    id = Column(Integer, primary_key=True)
    emp_id = Column(Integer, ForeignKey("employee.id"))
    day = Column(Date, nullable=False)
    shift_type = Column(String(20), nullable=False)
    __table_args__ = (UniqueConstraint("emp_id", "day", name="uix_emp_day"),)

class PositionShift(Base, HotelScoped):
    __tablename__ = "position_shift"
    id = Column(Integer, primary_key=True)
    department = Column(String)
    position = Column(String)
    shift_period = Column(String)
    shift_time = Column(String)

class ShiftTime(Base, HotelScoped):
    __tablename__ = "shift_times"
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey("positions.id"))
    period = Column(String)
    start = Column(Time)
    end = Column(Time)

class RoomActual(Base, HotelScoped):
    __tablename__ = "room_actual"
    id = Column(Integer, primary_key=True)
    kpi = Column(String)
    date = Column(Date)
    value = Column(Integer)

class RoomForecast(Base, HotelScoped):
    __tablename__ = "room_forecast"
    id = Column(Integer, primary_key=True)
    kpi = Column(String)
    date = Column(Date)
    value = Column(Integer)

class ScheduleAvailability(Base, HotelScoped):
    __tablename__ = "schedule_availability"
    id = Column(Integer, primary_key=True)
    emp_id = Column(Integer, ForeignKey("employee.id"))
    weekday = Column(String, nullable=False)
    availability = Column(String, nullable=False)
    employee = relationship("Employee", backref="availabilities")

class LaborStandard(Base, HotelScoped):
    __tablename__ = "labor_standards"
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey("positions.id"))
    metric = Column(String)
    standard = Column(Float)
    unit = Column(String, default="per FTE")

class RoomOTBPickup(Base, HotelScoped):
    __tablename__ = "room_otb_pickup"
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    kpi = Column(String)
    value = Column(Integer, default=0)

class OTBShift(Base, HotelScoped):
    __tablename__ = "otb_shift"
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer)
    date = Column(Date)
    hours = Column(Float)

class PlanningSummary(Base, HotelScoped):
    __tablename__ = "planning_summary"
    id = Column(Integer, primary_key=True)
    position = Column(String)
    date = Column(Date)
    scheduled_hours = Column(Float)
    fte = Column(Float)

class ProjectedHours(Base, HotelScoped):
    __tablename__ = "projected_hours"
    id = Column(Integer, primary_key=True)
    position = Column(String)
    date = Column(Date)
    otb_hours = Column(Float)
    fte = Column(Float)

class OTBHours(Base, HotelScoped):
    __tablename__ = "otb_hours"
    id = Column(Integer, primary_key=True)
    position = Column(String)
    date = Column(Date)
    otb_hours = Column(Float)
    fte = Column(Float)

class Actual(Base, HotelScoped):
    __tablename__ = "actual"
    id = Column(Integer, primary_key=True)
    emp_id = Column(Integer, ForeignKey("employee.id"))
    position_id = Column(Integer, ForeignKey("positions.id"))
    date = Column(Date, nullable=False)
    hours = Column(Float, default=0.0)
    ot_hours = Column(Float, default=0.0)
    reg_pay = Column(Float, default=0.0)
    ot_pay = Column(Float, default=0.0)
    source = Column(String, default="manual")

class Rooms(Base, HotelScoped):
    __tablename__ = "rooms"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True)
    occupied = Column(Integer, nullable=False)

# ────────────── INIT ──────────────
def init_db():
    Base.metadata.create_all(ENGINE)

with ENGINE.connect() as con:
    try:
        con.execute(text('ALTER TABLE employee ADD COLUMN emp_type TEXT NOT NULL DEFAULT "import";'))
    except Exception as e:
        print("Skipping emp_type column creation:", e)

init_db()

if __name__ == "__main__":
    init_db()
    print("✅ hotel_labor.db initialized")
