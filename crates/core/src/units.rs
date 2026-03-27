use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Trait: ToSI — all unit enums implement this for bidirectional conversion
// ---------------------------------------------------------------------------

/// Convert a value between a specific unit and its SI base unit.
pub trait ToSI {
    /// Convert `value` expressed in `self` to SI base units.
    fn to_si(&self, value: f64) -> f64;
    /// Convert `value` expressed in SI base units to `self`.
    fn from_si(&self, value: f64) -> f64;
}

// ---------------------------------------------------------------------------
// Length (SI base: meters)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LengthUnit {
    Meters,
    Feet,
    Centimeters,
    Inches,
    Kilometers,
    Miles,
}

impl ToSI for LengthUnit {
    fn to_si(&self, value: f64) -> f64 {
        match self {
            Self::Meters => value,
            Self::Feet => value * 0.3048,
            Self::Centimeters => value * 0.01,
            Self::Inches => value * 0.0254,
            Self::Kilometers => value * 1000.0,
            Self::Miles => value * 1609.344,
        }
    }

    fn from_si(&self, value: f64) -> f64 {
        match self {
            Self::Meters => value,
            Self::Feet => value / 0.3048,
            Self::Centimeters => value / 0.01,
            Self::Inches => value / 0.0254,
            Self::Kilometers => value / 1000.0,
            Self::Miles => value / 1609.344,
        }
    }
}

impl fmt::Display for LengthUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meters => write!(f, "m"),
            Self::Feet => write!(f, "ft"),
            Self::Centimeters => write!(f, "cm"),
            Self::Inches => write!(f, "in"),
            Self::Kilometers => write!(f, "km"),
            Self::Miles => write!(f, "mi"),
        }
    }
}

// ---------------------------------------------------------------------------
// Velocity (SI base: m/s)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VelocityUnit {
    MetersPerSec,
    FeetPerSec,
    Mph,
    Kmh,
    KilometersPerSec,
    Knots,
}

impl ToSI for VelocityUnit {
    fn to_si(&self, value: f64) -> f64 {
        match self {
            Self::MetersPerSec => value,
            Self::FeetPerSec => value * 0.3048,
            Self::Mph => value * 0.44704,
            Self::Kmh => value / 3.6,
            Self::KilometersPerSec => value * 1000.0,
            Self::Knots => value * 0.514444,
        }
    }

    fn from_si(&self, value: f64) -> f64 {
        match self {
            Self::MetersPerSec => value,
            Self::FeetPerSec => value / 0.3048,
            Self::Mph => value / 0.44704,
            Self::Kmh => value * 3.6,
            Self::KilometersPerSec => value / 1000.0,
            Self::Knots => value / 0.514444,
        }
    }
}

impl fmt::Display for VelocityUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetersPerSec => write!(f, "m/s"),
            Self::FeetPerSec => write!(f, "ft/s"),
            Self::Mph => write!(f, "mph"),
            Self::Kmh => write!(f, "km/h"),
            Self::KilometersPerSec => write!(f, "km/s"),
            Self::Knots => write!(f, "kn"),
        }
    }
}

// ---------------------------------------------------------------------------
// Angle (SI base: radians)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AngleUnit {
    Radians,
    Degrees,
}

impl ToSI for AngleUnit {
    fn to_si(&self, value: f64) -> f64 {
        match self {
            Self::Radians => value,
            Self::Degrees => value.to_radians(),
        }
    }

    fn from_si(&self, value: f64) -> f64 {
        match self {
            Self::Radians => value,
            Self::Degrees => value.to_degrees(),
        }
    }
}

impl fmt::Display for AngleUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Radians => write!(f, "rad"),
            Self::Degrees => write!(f, "°"),
        }
    }
}

// ---------------------------------------------------------------------------
// Mass (SI base: kilograms)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MassUnit {
    Kilograms,
    Grams,
    Pounds,
    Tons,
}

impl ToSI for MassUnit {
    fn to_si(&self, value: f64) -> f64 {
        match self {
            Self::Kilograms => value,
            Self::Grams => value * 0.001,
            Self::Pounds => value * 0.453_592_37,
            Self::Tons => value * 907.184_74,
        }
    }

    fn from_si(&self, value: f64) -> f64 {
        match self {
            Self::Kilograms => value,
            Self::Grams => value / 0.001,
            Self::Pounds => value / 0.453_592_37,
            Self::Tons => value / 907.184_74,
        }
    }
}

impl fmt::Display for MassUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kilograms => write!(f, "kg"),
            Self::Grams => write!(f, "g"),
            Self::Pounds => write!(f, "lb"),
            Self::Tons => write!(f, "ton"),
        }
    }
}

// ---------------------------------------------------------------------------
// Acceleration (SI base: m/s²)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccelerationUnit {
    MetersPerSecSq,
    FeetPerSecSq,
    StandardG,
}

impl ToSI for AccelerationUnit {
    fn to_si(&self, value: f64) -> f64 {
        match self {
            Self::MetersPerSecSq => value,
            Self::FeetPerSecSq => value * 0.3048,
            Self::StandardG => value * 9.806_65,
        }
    }

    fn from_si(&self, value: f64) -> f64 {
        match self {
            Self::MetersPerSecSq => value,
            Self::FeetPerSecSq => value / 0.3048,
            Self::StandardG => value / 9.806_65,
        }
    }
}

impl fmt::Display for AccelerationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetersPerSecSq => write!(f, "m/s²"),
            Self::FeetPerSecSq => write!(f, "ft/s²"),
            Self::StandardG => write!(f, "g"),
        }
    }
}

// ---------------------------------------------------------------------------
// Area (SI base: m²)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AreaUnit {
    SquareMeters,
    SquareFeet,
    SquareCentimeters,
    SquareInches,
}

impl ToSI for AreaUnit {
    fn to_si(&self, value: f64) -> f64 {
        match self {
            Self::SquareMeters => value,
            Self::SquareFeet => value * 0.092_903_04,
            Self::SquareCentimeters => value * 0.0001,
            Self::SquareInches => value * 0.000_645_16,
        }
    }

    fn from_si(&self, value: f64) -> f64 {
        match self {
            Self::SquareMeters => value,
            Self::SquareFeet => value / 0.092_903_04,
            Self::SquareCentimeters => value / 0.0001,
            Self::SquareInches => value / 0.000_645_16,
        }
    }
}

impl fmt::Display for AreaUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SquareMeters => write!(f, "m²"),
            Self::SquareFeet => write!(f, "ft²"),
            Self::SquareCentimeters => write!(f, "cm²"),
            Self::SquareInches => write!(f, "in²"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn assert_approx(a: f64, b: f64) {
        assert!(
            (a - b).abs() < EPSILON,
            "expected {b}, got {a}, diff = {}",
            (a - b).abs()
        );
    }

    #[test]
    fn length_roundtrip() {
        let units = [
            LengthUnit::Meters,
            LengthUnit::Feet,
            LengthUnit::Centimeters,
            LengthUnit::Inches,
            LengthUnit::Kilometers,
            LengthUnit::Miles,
        ];
        for unit in &units {
            let original = 42.0;
            let si = unit.to_si(original);
            let back = unit.from_si(si);
            assert_approx(back, original);
        }
    }

    #[test]
    fn velocity_known_values() {
        assert_approx(VelocityUnit::Mph.to_si(60.0), 26.8224);
        assert_approx(VelocityUnit::Kmh.to_si(100.0), 27.777_777_777_777_78);
        assert_approx(VelocityUnit::Knots.to_si(1.0), 0.514444);
    }

    #[test]
    fn angle_conversion() {
        assert_approx(AngleUnit::Degrees.to_si(180.0), std::f64::consts::PI);
        assert_approx(AngleUnit::Radians.to_si(std::f64::consts::PI), std::f64::consts::PI);
    }

    #[test]
    fn mass_roundtrip() {
        let units = [
            MassUnit::Kilograms,
            MassUnit::Grams,
            MassUnit::Pounds,
            MassUnit::Tons,
        ];
        for unit in &units {
            let original = 100.0;
            let si = unit.to_si(original);
            let back = unit.from_si(si);
            assert_approx(back, original);
        }
    }

    #[test]
    fn acceleration_standard_g() {
        assert_approx(AccelerationUnit::StandardG.to_si(1.0), 9.80665);
    }

    #[test]
    fn area_roundtrip() {
        let units = [
            AreaUnit::SquareMeters,
            AreaUnit::SquareFeet,
            AreaUnit::SquareCentimeters,
            AreaUnit::SquareInches,
        ];
        for unit in &units {
            let original = 55.5;
            let si = unit.to_si(original);
            let back = unit.from_si(si);
            assert_approx(back, original);
        }
    }
}
