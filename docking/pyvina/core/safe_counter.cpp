#include "safe_counter.hpp"

template <typename T>
void safe_counter<T>::init(const T z)
{
	n = z;
	i = 0;
}

template <typename T>
void safe_counter<T>::increment()
{
	lock_guard<mutex> guard(m);
	if (++i == n) cv.notify_one();
}

template <typename T>
void safe_counter<T>::wait()
{
	unique_lock<mutex> lock(m);
	if (i < n) cv.wait(lock);
}

template class safe_counter<size_t>;
