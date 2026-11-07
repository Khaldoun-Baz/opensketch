//! Length units for display and for parsing typed values (Measurements box).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Units {
    Millimeters,
    Centimeters,
    Meters,
    Inches,
    Feet,
}

impl Units {
    pub const ALL: [Units; 5] = [Units::Millimeters, Units::Centimeters, Units::Meters, Units::Inches, Units::Feet];

    pub fn label(self) -> &'static str {
        match self {
            Units::Millimeters => "Millimeters",
            Units::Centimeters => "Centimeters",
            Units::Meters => "Meters",
            Units::Inches => "Inches",
            Units::Feet => "Feet",
        }
    }

    /// Meters per unit.
    pub fn factor(self) -> f32 {
        match self {
            Units::Millimeters => 0.001,
            Units::Centimeters => 0.01,
            Units::Meters => 1.0,
            Units::Inches => 0.0254,
            Units::Feet => 0.3048,
        }
    }

    pub fn format(self, meters: f32) -> String {
        let v = meters / self.factor();
        match self {
            Units::Millimeters => format!("{v:.0} mm"),
            Units::Centimeters => format!("{v:.1} cm"),
            Units::Meters => format!("{v:.3} m"),
            Units::Inches => format!("{v:.2}\""),
            Units::Feet => format!("{v:.2}'"),
        }
    }

    pub fn format_area(self, m2: f32) -> String {
        let f = self.factor();
        let v = m2 / (f * f);
        match self {
            Units::Millimeters => format!("{v:.0} mm²"),
            Units::Centimeters => format!("{v:.1} cm²"),
            Units::Meters => format!("{v:.3} m²"),
            Units::Inches => format!("{v:.2} in²"),
            Units::Feet => format!("{v:.2} ft²"),
        }
    }

    pub fn format_volume(self, m3: f32) -> String {
        let f = self.factor();
        let v = m3 / (f * f * f);
        match self {
            Units::Millimeters => format!("{v:.0} mm³"),
            Units::Centimeters => format!("{v:.1} cm³"),
            Units::Meters => format!("{v:.3} m³"),
            Units::Inches => format!("{v:.2} in³"),
            Units::Feet => format!("{v:.2} ft³"),
        }
    }

    /// Parses "2.5", "250mm", "25 cm", "3m", "12\"", "4'" into meters.
    /// Bare numbers use the current display unit.
    pub fn parse(self, s: &str) -> Option<f32> {
        let s = s.trim().to_lowercase();
        let suffixes: [(&str, f32); 7] =
            [("mm", 0.001), ("cm", 0.01), ("in", 0.0254), ("ft", 0.3048), ("m", 1.0), ("\"", 0.0254), ("'", 0.3048)];
        for (suf, f) in suffixes {
            if let Some(num) = s.strip_suffix(suf) {
                return num.trim().parse::<f32>().ok().map(|v| v * f);
            }
        }
        s.parse::<f32>().ok().map(|v| v * self.factor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_units() {
        assert_eq!(Units::Meters.parse("2.5"), Some(2.5));
        assert!((Units::Meters.parse("250mm").unwrap() - 0.25).abs() < 1e-6);
        assert!((Units::Millimeters.parse("300").unwrap() - 0.3).abs() < 1e-6);
        assert!((Units::Meters.parse("12\"").unwrap() - 0.3048).abs() < 1e-5);
        assert_eq!(Units::Meters.parse("abc"), None);
    }
}
