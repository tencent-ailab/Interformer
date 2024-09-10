#pragma once
#ifndef IDOCK_SAFE_COUNTER_HPP
#define IDOCK_SAFE_COUNTER_HPP

#include <condition_variable>
using namespace std;

//! Represents a thread safe counter.
template <typename T>
class safe_counter
{
public:
	//! Initializes the counter to 0 and its expected hit value to z.
	void init(const T z);

	//! Increments the counter by 1 in a thread safe manner, and wakes up the calling thread waiting on the internal mutex.
	void increment();

	//! Waits until the counter reaches its expected hit value.
	void wait();
private:
	mutex m;
	condition_variable cv;
	T n; //!< Expected hit value.
	T i; //!< Counter value.
};

#endif
