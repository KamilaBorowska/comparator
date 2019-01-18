//! Comparator combinators.

use core::cmp::Ordering;
use core::fmt;
use Comparator;

/// A comparator that reverse its underlying comparator.
#[derive(Copy, Clone, Debug)]
pub struct Reversed<T>(pub(crate) T);

impl<T, U> Comparator<U> for Reversed<T>
where
    T: Comparator<U>,
    U: ?Sized,
{
    fn compare(&self, a: &U, b: &U) -> Ordering {
        self.0.compare(a, b).reverse()
    }
}

#[derive(Copy, Clone, Debug)]
/// A comparator that chains another comparator.
pub struct ThenComparing<T, U>(pub(crate) T, pub(crate) U);

impl<T, U, W> Comparator<W> for ThenComparing<T, U>
where
    T: Comparator<W>,
    U: Comparator<W>,
    W: ?Sized,
{
    fn compare(&self, a: &W, b: &W) -> Ordering {
        self.0.compare(a, b).then_with(|| self.1.compare(a, b))
    }
}

#[derive(Copy, Clone)]
/// A comparator that chains comparison function.
pub struct ThenComparingByKey<T, F> {
    pub(crate) comparator: T,
    pub(crate) f: F,
}

impl<T, F> fmt::Debug for ThenComparingByKey<T, F>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ThenComparingByKey")
            .field("comparator", &self.comparator)
            .finish()
    }
}

impl<T, F, U, W> Comparator<U> for ThenComparingByKey<T, F>
where
    T: Comparator<U>,
    F: Fn(&U) -> W,
    U: ?Sized,
    W: Ord,
{
    fn compare(&self, a: &U, b: &U) -> Ordering {
        self.comparator
            .compare(a, b)
            .then_with(|| (self.f)(&a).cmp(&(self.f)(&b)))
    }
}

/// Create a combinator comparing by key.
pub fn comparing<F, T, U>(f: F) -> Comparing<F>
where
    F: Fn(&T) -> U,
    T: ?Sized,
    U: Ord,
{
    Comparing { f }
}

/// A comparator that compares using its key function.
#[derive(Clone, Copy)]
pub struct Comparing<F> {
    f: F,
}

impl<F> fmt::Debug for Comparing<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Comparing").finish()
    }
}

impl<F, T, U> Comparator<T> for Comparing<F>
where
    F: Fn(&T) -> U,
    T: ?Sized,
    U: Ord,
{
    fn compare(&self, a: &T, b: &T) -> Ordering {
        (self.f)(a).cmp(&(self.f)(b))
    }
}

/// Creates a natural ordering comparator.
pub fn natural_order() -> NaturalOrder {
    NaturalOrder { _priv: () }
}

/// A natural ordering comparator.
#[derive(Copy, Clone, Default)]
pub struct NaturalOrder {
    _priv: (),
}

impl fmt::Debug for NaturalOrder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NaturalOrder").finish()
    }
}

impl<T> Comparator<T> for NaturalOrder
where
    T: ?Sized + Ord,
{
    fn compare(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

/// Creates a reverse ordering comparator.
pub fn reverse_order() -> ReverseOrder {
    ReverseOrder { _priv: () }
}

/// Reverse ordering comparator.
#[derive(Copy, Clone, Default)]
pub struct ReverseOrder {
    _priv: (),
}

impl fmt::Debug for ReverseOrder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ReverseOrder").finish()
    }
}

impl<T> Comparator<T> for ReverseOrder
where
    T: ?Sized + Ord,
{
    fn compare(&self, a: &T, b: &T) -> Ordering {
        b.cmp(a)
    }
}
