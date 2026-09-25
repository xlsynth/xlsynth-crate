// SPDX-License-Identifier: Apache-2.0

//! Structural types for XLS IR values.

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ArrayTypeData {
    pub element_type: Box<Type>,
    pub element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Type {
    Token,
    Bits(usize),
    Tuple(Vec<Box<Type>>),
    Array(ArrayTypeData),
}

/// Represents an interval of the form `[start, limit)` i.e. inclusive start
/// exclusive limit.
pub struct StartAndLimit {
    pub start: usize,
    pub limit: usize,
}

impl Type {
    pub fn new_array(element_type: Type, element_count: usize) -> Self {
        Type::Array(ArrayTypeData {
            element_type: Box::new(element_type),
            element_count,
        })
    }

    pub fn nil() -> Self {
        Type::Tuple(vec![])
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Type::Tuple(types) if types.is_empty())
    }

    pub fn bit_count(&self) -> usize {
        match self {
            Type::Token => 0,
            Type::Bits(width) => *width,
            Type::Tuple(types) => types.iter().map(|t| t.bit_count()).sum(),
            Type::Array(ArrayTypeData {
                element_type,
                element_count,
            }) => element_type.bit_count() * element_count,
        }
    }

    /// Returns the flattened width, or `None` if aggregate size arithmetic
    /// overflows.
    pub fn checked_bit_count(&self) -> Option<usize> {
        match self {
            Type::Token => Some(0),
            Type::Bits(width) => Some(*width),
            Type::Tuple(types) => types.iter().try_fold(0usize, |width, ty| {
                width.checked_add(ty.checked_bit_count()?)
            }),
            Type::Array(array) => array
                .element_type
                .checked_bit_count()?
                .checked_mul(array.element_count),
        }
    }

    /// Returns the start and limit bits for our bitwise representation of a
    /// tuple access at the given index.
    ///
    /// E.g. consider `tuple(a, b, c)`, we represent this in a bit vector as:
    /// `a_msb, ..., a_lsb, b_msb, ..., b_lsb, c_msb, ..., c_lsb` where c_lsb is
    /// the least significant bit of the overall bit vector. That means to
    /// access index `i` we have to slice out all the members that come
    /// after it; e.g. if we want to access b we have to skip `c`
    /// least significant bits.
    pub fn tuple_get_flat_bit_slice_for_index(
        &self,
        index: usize,
    ) -> Result<StartAndLimit, String> {
        match self {
            Type::Tuple(types) => {
                let bits_after_index = types[index + 1..].iter().map(|t| t.bit_count()).sum();
                let limit = types[index].bit_count() + bits_after_index;
                Ok(StartAndLimit {
                    start: bits_after_index,
                    limit,
                })
            }
            _ => Err(format!(
                "Attempted to get bit slice for non-tuple type: {:?}",
                self
            )),
        }
    }

    pub fn get_array_element_type(&self) -> &Type {
        match self {
            Type::Array(ArrayTypeData { element_type, .. }) => element_type,
            _ => panic!(
                "Attempted to get array element type for non-array type: {:?}",
                self
            ),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Token => write!(f, "token"),
            Type::Bits(width) => write!(f, "bits[{}]", width),
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            Type::Array(ArrayTypeData {
                element_type,
                element_count,
            }) => {
                write!(f, "{}", element_type)?;
                write!(f, "[{}]", element_count)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_type_width_handles_aggregates_and_overflow() {
        assert_eq!(Type::Token.checked_bit_count(), Some(0));
        assert_eq!(Type::nil().checked_bit_count(), Some(0));
        assert_eq!(Type::Bits(usize::MAX).checked_bit_count(), Some(usize::MAX));
        let fields = Type::Tuple(vec![Box::new(Type::Bits(65)), Box::new(Type::Bits(129))]);
        assert_eq!(Type::new_array(fields, 3).checked_bit_count(), Some(582));
        assert_eq!(
            Type::new_array(Type::Bits(usize::MAX), 2).checked_bit_count(),
            None
        );
        assert_eq!(
            Type::Tuple(vec![
                Box::new(Type::Bits(usize::MAX)),
                Box::new(Type::Bits(1))
            ])
            .checked_bit_count(),
            None
        );
    }
}
