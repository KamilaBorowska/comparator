//! A Java-like Comparator type.

#![deny(clippy::all, clippy::cargo, missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
use std as core;

#[cfg(feature = "std")]
pub mod collections;
pub mod combinators;

pub use combinators::{comparing, natural_order, reverse_order};
use combinators::{Reversed, ThenComparing, ThenComparingByKey};
use core::cmp::Ordering;

/// An interface for dealing with comparators.
pub trait Comparator<T> {
    /// Compares its two arguments for order.
    fn compare(&self, a: &T, b: &T) -> Ordering;

    /// Reverses the `Comparator`.
    fn reversed(self) -> Reversed<Self>
    where
        Self: Sized,
    {
        Reversed(self)
    }

    /// Chains two comparators.
    ///
    /// Returns the result of the first comparator when it's not `Equal`.
    /// Otherwise returns the result of the second comparator.
    ///
    /// # Examples
    ///
    /// ```
    /// use comparator::{as_fn, comparing, natural_order, Comparator};
    /// let mut v = [4, 2, 6, 3, 1, 5, 8, 7];
    /// // Groups numbers by even-ness and sorts them.
    /// v.sort_by(as_fn(comparing(|i| i % 2).then_comparing(natural_order())));
    /// assert_eq!(v, [2, 4, 6, 8, 1, 3, 5, 7]);
    /// ```
    fn then_comparing<U>(self, other: U) -> ThenComparing<Self, U>
    where
        Self: Sized,
        U: Comparator<T>,
    {
        ThenComparing(self, other)
    }

    /// Chains a comparator with a key function.
    ///
    /// Returns the result of the first comparator when it's not `Equal`.
    /// Otherwise calls `f` and returns the result.
    fn then_comparing_by_key<F, U>(self, f: F) -> ThenComparingByKey<Self, F>
    where
        Self: Sized,
        F: Fn(&T) -> U,
        U: Ord,
    {
        ThenComparingByKey {
            comparator: self,
            f,
        }
    }
}

impl<F, T> Comparator<T> for F
where
    F: Fn(&T, &T) -> Ordering,
{
    fn compare(&self, a: &T, b: &T) -> Ordering {
        self(a, b)
    }
}

/// Convert a comparator to a comparator function.
///
/// This is not a member function, as in stable Rust it's not possible
/// to implement `Fn` trait manually, forcing usage of `impl Trait`
/// return type. Currently it is not allowed for trait methods to return
/// `impl Trait`s.
///
/// # Examples
///
/// ```
/// use comparator::{as_fn, reverse_order};
/// let mut v = [5, 4, 1, 3, 2];
/// v.sort_by(as_fn(reverse_order()));
/// assert_eq!(v, [5, 4, 3, 2, 1]);
/// ```
pub fn as_fn<T>(comparator: impl Comparator<T>) -> impl Fn(&T, &T) -> Ordering {
    move |a, b| comparator.compare(a, b)
}
