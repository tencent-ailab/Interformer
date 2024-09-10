#pragma once
#ifndef IO_SERVICE_POOL_HPP
#define IO_SERVICE_POOL_HPP

#include <boost/thread/future.hpp>
#include <boost/asio/io_service.hpp>
using namespace std;
using namespace boost::asio;

//! Represents a pool of threads concurrently an instance of io service.
class io_service_pool : public io_service, public vector<boost::unique_future<void>>
{
public:
	//! Creates a number of threads to listen to the post event of an io service.
	explicit io_service_pool(const size_t num_threads);

	//! Waits for all the posted work and created threads to complete, and propagates thrown exceptions if any.
	void wait();
private:
	unique_ptr<work> w; //!< An io service work object, resetting which to nullptr signals the io service to stop receiving additional work.
};

#endif
