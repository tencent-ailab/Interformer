#include <numeric>
#include <algorithm>
#include "random_forest.hpp"

node::node() : children{}
{
}

void tree::train(const size_t mtry, const function<double()> u01)
{
	// Create bootstrap samples with replacement.
	reserve((ns << 1) - 1);
	emplace_back();
	node& root = front();
	root.samples.resize(ns);
	for (size_t& s : root.samples)
	{
		s = static_cast<size_t>(u01() * ns);
	}

	// Populate nodes.
	for (size_t k = 0; k < size(); ++k)
	{
		node& n = (*this)[k];

		// Evaluate node y and purity.
		double sum = 0;
		for (const size_t s : n.samples) sum += y[s];
		n.y = sum / n.samples.size();
		n.p = sum * n.y; // = n.y * n.y * n.samples.size() = sum * sum / n.samples.size().

		// Do not split the node if it contains too few samples.
		if (n.samples.size() <= 5) continue;

		// Find the best split that has the highest increase in node purity.
		double bestChildNodePurity = n.p;
		array<size_t, nv> mind;
		iota(mind.begin(), mind.end(), 0);
		for (size_t i = 0; i < mtry; ++i)
		{
			// Randomly select a variable without replacement.
			const size_t j = static_cast<size_t>(u01() * (nv - i));
			const size_t v = mind[j];
			mind[j] = mind[nv - i - 1];

			// Sort the samples in ascending order of the selected variable.
			vector<size_t> ncase(n.samples.size());
			iota(ncase.begin(), ncase.end(), 0);
			sort(ncase.begin(), ncase.end(), [&n, v](const size_t val1, const size_t val2)
			{
				return x[n.samples[val1]][v] < x[n.samples[val2]][v];
			});

			// Search through the gaps in the selected variable.
			double suml = 0;
			double sumr = sum;
			size_t popl = 0;
			size_t popr = n.samples.size();
			for (size_t j = 0; j < n.samples.size() - 1; ++j)
			{
				const double d = y[n.samples[ncase[j]]];
				suml += d;
				sumr -= d;
				++popl;
				--popr;
				if (x[n.samples[ncase[j]]][v] == x[n.samples[ncase[j+1]]][v]) continue;
				const double curChildNodePurity = (suml * suml / popl) + (sumr * sumr / popr);
				if (curChildNodePurity > bestChildNodePurity)
				{
					bestChildNodePurity = curChildNodePurity;
					n.var = v;
					n.val = (x[n.samples[ncase[j]]][v] + x[n.samples[ncase[j+1]]][v]) * 0.5;
				}
			}
		}

		// Do not split the node if purity does not increase.
		if (bestChildNodePurity == n.p) continue;

		// Create two child nodes and distribute samples.
		n.children[0] = size();
		emplace_back();
		n.children[1] = size();
		emplace_back();
		for (const size_t s : n.samples)
		{
			(*this)[n.children[x[s][n.var] > n.val]].samples.push_back(s);
		}
	}
}

double tree::operator()(const array<double, nv>& x) const
{
	size_t k;
	for (k = 0; (*this)[k].children[0]; k = (*this)[k].children[x[(*this)[k].var] > (*this)[k].val]);
	return (*this)[k].y;
}

void tree::clear()
{
	for (node& n : *this)
	{
		n.samples.clear();
	}
}

forest::forest(const size_t nt, const size_t seed) : vector<tree>(nt), u01_s([&]()
{
	lock_guard<mutex> guard(m);
	return uniform_01(rng);
}), nt_inv(1.0 / nt), rng(seed), uniform_01(0, 1)
{
}

double forest::operator()(const array<double, tree::nv>& x) const
{
	double y = 0;
	for (const tree& t : *this)
	{
		y += t(x);
	}
	return y *= nt_inv;
}

void forest::clear()
{
	for (tree& t : *this)
	{
		t.clear();
	}
}
