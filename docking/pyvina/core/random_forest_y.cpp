#include "random_forest.hpp"

const array<double, tree::ns> tree::y
{{
	2.00,
	2.00,
	2.00,
	2.00,
	2.00,
	2.00,
	2.05,
	2.07,
	2.10,
	2.12,
	2.12,
	2.16,
	2.17,
	2.18,
	2.19,
	2.19,
	2.19,
	2.21,
	2.21,
	2.22,
	2.22,
	2.22,
	2.23,
	2.23,
	2.23,
	2.23,
	2.26,
	2.26,
	2.26,
	2.27,
	2.27,
	2.28,
	2.28,
	2.29,
	2.30,
	2.30,
	2.30,
	2.30,
	2.30,
	2.30,
	2.30,
	2.31,
	2.34,
	2.35,
	2.36,
	2.37,
	2.37,
	2.38,
	2.39,
	2.39,
	2.40,
	2.40,
	2.44,
	2.44,
	2.44,
	2.44,
	2.46,
	2.46,
	2.47,
	2.47,
	2.48,
	2.49,
	2.49,
	2.50,
	2.51,
	2.52,
	2.52,
	2.52,
	2.52,
	2.52,
	2.53,
	2.55,
	2.55,
	2.57,
	2.57,
	2.59,
	2.59,
	2.60,
	2.60,
	2.62,
	2.62,
	2.63,
	2.64,
	2.66,
	2.67,
	2.68,
	2.68,
	2.69,
	2.70,
	2.70,
	2.70,
	2.72,
	2.72,
	2.74,
	2.74,
	2.74,
	2.74,
	2.75,
	2.76,
	2.77,
	2.77,
	2.77,
	2.80,
	2.81,
	2.82,
	2.82,
	2.82,
	2.82,
	2.84,
	2.84,
	2.85,
	2.86,
	2.86,
	2.86,
	2.89,
	2.89,
	2.89,
	2.89,
	2.89,
	2.89,
	2.89,
	2.89,
	2.90,
	2.92,
	2.92,
	2.92,
	2.92,
	2.92,
	2.92,
	2.92,
	2.92,
	2.93,
	2.94,
	2.96,
	2.96,
	2.96,
	2.96,
	2.96,
	2.96,
	2.96,
	2.96,
	2.97,
	2.97,
	2.98,
	2.98,
	2.98,
	3.00,
	3.00,
	3.00,
	3.00,
	3.00,
	3.00,
	3.00,
	3.00,
	3.00,
	3.00,
	3.00,
	3.01,
	3.01,
	3.02,
	3.03,
	3.03,
	3.03,
	3.03,
	3.04,
	3.04,
	3.04,
	3.04,
	3.04,
	3.05,
	3.05,
	3.05,
	3.05,
	3.06,
	3.07,
	3.07,
	3.08,
	3.10,
	3.10,
	3.10,
	3.10,
	3.10,
	3.10,
	3.11,
	3.12,
	3.12,
	3.13,
	3.14,
	3.14,
	3.15,
	3.15,
	3.15,
	3.15,
	3.16,
	3.16,
	3.18,
	3.18,
	3.18,
	3.19,
	3.19,
	3.19,
	3.21,
	3.22,
	3.22,
	3.22,
	3.22,
	3.22,
	3.22,
	3.22,
	3.22,
	3.22,
	3.22,
	3.23,
	3.23,
	3.23,
	3.24,
	3.24,
	3.25,
	3.25,
	3.25,
	3.26,
	3.26,
	3.27,
	3.28,
	3.28,
	3.28,
	3.28,
	3.28,
	3.30,
	3.30,
	3.30,
	3.30,
	3.30,
	3.30,
	3.30,
	3.30,
	3.30,
	3.30,
	3.30,
	3.31,
	3.31,
	3.32,
	3.32,
	3.32,
	3.32,
	3.33,
	3.33,
	3.33,
	3.33,
	3.33,
	3.33,
	3.33,
	3.34,
	3.34,
	3.34,
	3.35,
	3.35,
	3.35,
	3.36,
	3.36,
	3.37,
	3.37,
	3.37,
	3.37,
	3.37,
	3.37,
	3.38,
	3.38,
	3.38,
	3.39,
	3.39,
	3.39,
	3.40,
	3.40,
	3.40,
	3.40,
	3.40,
	3.40,
	3.40,
	3.40,
	3.40,
	3.40,
	3.41,
	3.42,
	3.42,
	3.42,
	3.42,
	3.42,
	3.42,
	3.43,
	3.43,
	3.44,
	3.45,
	3.46,
	3.46,
	3.46,
	3.46,
	3.47,
	3.47,
	3.47,
	3.48,
	3.48,
	3.48,
	3.48,
	3.48,
	3.49,
	3.49,
	3.49,
	3.49,
	3.50,
	3.51,
	3.52,
	3.52,
	3.52,
	3.52,
	3.52,
	3.52,
	3.52,
	3.54,
	3.54,
	3.55,
	3.55,
	3.55,
	3.55,
	3.57,
	3.57,
	3.57,
	3.57,
	3.57,
	3.57,
	3.58,
	3.59,
	3.59,
	3.59,
	3.60,
	3.60,
	3.60,
	3.60,
	3.60,
	3.60,
	3.60,
	3.60,
	3.60,
	3.60,
	3.61,
	3.61,
	3.62,
	3.62,
	3.63,
	3.64,
	3.64,
	3.64,
	3.64,
	3.65,
	3.66,
	3.66,
	3.66,
	3.66,
	3.66,
	3.66,
	3.66,
	3.66,
	3.67,
	3.67,
	3.68,
	3.68,
	3.68,
	3.68,
	3.68,
	3.68,
	3.68,
	3.69,
	3.69,
	3.69,
	3.70,
	3.70,
	3.70,
	3.70,
	3.70,
	3.70,
	3.70,
	3.70,
	3.71,
	3.71,
	3.71,
	3.72,
	3.72,
	3.72,
	3.72,
	3.72,
	3.72,
	3.72,
	3.73,
	3.73,
	3.74,
	3.74,
	3.75,
	3.75,
	3.76,
	3.76,
	3.76,
	3.77,
	3.77,
	3.77,
	3.77,
	3.78,
	3.79,
	3.80,
	3.80,
	3.80,
	3.80,
	3.80,
	3.80,
	3.81,
	3.81,
	3.82,
	3.82,
	3.82,
	3.82,
	3.82,
	3.82,
	3.84,
	3.84,
	3.84,
	3.85,
	3.85,
	3.85,
	3.85,
	3.86,
	3.86,
	3.86,
	3.87,
	3.87,
	3.87,
	3.87,
	3.87,
	3.87,
	3.88,
	3.88,
	3.89,
	3.89,
	3.89,
	3.89,
	3.89,
	3.89,
	3.89,
	3.90,
	3.90,
	3.90,
	3.90,
	3.90,
	3.90,
	3.90,
	3.90,
	3.90,
	3.90,
	3.90,
	3.91,
	3.92,
	3.92,
	3.92,
	3.93,
	3.93,
	3.93,
	3.94,
	3.94,
	3.95,
	3.96,
	3.96,
	3.96,
	3.96,
	3.96,
	3.96,
	3.97,
	3.97,
	3.98,
	3.98,
	3.98,
	3.99,
	3.99,
	3.99,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.00,
	4.01,
	4.01,
	4.01,
	4.02,
	4.02,
	4.03,
	4.03,
	4.03,
	4.03,
	4.03,
	4.03,
	4.04,
	4.04,
	4.05,
	4.05,
	4.05,
	4.05,
	4.05,
	4.05,
	4.05,
	4.06,
	4.06,
	4.06,
	4.07,
	4.07,
	4.08,
	4.08,
	4.08,
	4.09,
	4.09,
	4.09,
	4.09,
	4.09,
	4.09,
	4.10,
	4.10,
	4.10,
	4.10,
	4.10,
	4.11,
	4.11,
	4.11,
	4.11,
	4.11,
	4.11,
	4.12,
	4.12,
	4.12,
	4.12,
	4.12,
	4.12,
	4.13,
	4.13,
	4.14,
	4.14,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.15,
	4.16,
	4.16,
	4.17,
	4.17,
	4.17,
	4.18,
	4.18,
	4.19,
	4.19,
	4.19,
	4.19,
	4.19,
	4.19,
	4.19,
	4.19,
	4.20,
	4.20,
	4.20,
	4.20,
	4.20,
	4.20,
	4.20,
	4.21,
	4.21,
	4.21,
	4.22,
	4.22,
	4.22,
	4.22,
	4.22,
	4.22,
	4.22,
	4.22,
	4.22,
	4.23,
	4.23,
	4.23,
	4.24,
	4.24,
	4.25,
	4.26,
	4.26,
	4.27,
	4.28,
	4.28,
	4.28,
	4.28,
	4.29,
	4.29,
	4.29,
	4.29,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.30,
	4.31,
	4.31,
	4.31,
	4.32,
	4.32,
	4.32,
	4.32,
	4.33,
	4.33,
	4.33,
	4.33,
	4.34,
	4.34,
	4.34,
	4.35,
	4.35,
	4.35,
	4.35,
	4.36,
	4.36,
	4.36,
	4.36,
	4.37,
	4.37,
	4.38,
	4.38,
	4.38,
	4.38,
	4.39,
	4.39,
	4.39,
	4.40,
	4.40,
	4.40,
	4.40,
	4.40,
	4.40,
	4.40,
	4.41,
	4.41,
	4.41,
	4.41,
	4.41,
	4.41,
	4.41,
	4.41,
	4.42,
	4.42,
	4.42,
	4.42,
	4.43,
	4.43,
	4.43,
	4.43,
	4.43,
	4.43,
	4.43,
	4.43,
	4.43,
	4.43,
	4.43,
	4.44,
	4.44,
	4.45,
	4.46,
	4.46,
	4.47,
	4.47,
	4.47,
	4.48,
	4.48,
	4.48,
	4.48,
	4.48,
	4.48,
	4.48,
	4.49,
	4.49,
	4.49,
	4.49,
	4.49,
	4.49,
	4.49,
	4.50,
	4.50,
	4.50,
	4.51,
	4.51,
	4.51,
	4.51,
	4.51,
	4.51,
	4.52,
	4.52,
	4.52,
	4.52,
	4.52,
	4.52,
	4.52,
	4.52,
	4.52,
	4.52,
	4.52,
	4.53,
	4.54,
	4.54,
	4.54,
	4.54,
	4.54,
	4.54,
	4.54,
	4.54,
	4.55,
	4.55,
	4.55,
	4.55,
	4.55,
	4.55,
	4.55,
	4.57,
	4.57,
	4.58,
	4.59,
	4.59,
	4.59,
	4.59,
	4.59,
	4.59,
	4.59,
	4.59,
	4.59,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.60,
	4.61,
	4.61,
	4.61,
	4.61,
	4.62,
	4.62,
	4.62,
	4.62,
	4.62,
	4.62,
	4.62,
	4.62,
	4.62,
	4.62,
	4.63,
	4.63,
	4.64,
	4.65,
	4.65,
	4.65,
	4.66,
	4.66,
	4.66,
	4.66,
	4.66,
	4.66,
	4.66,
	4.66,
	4.67,
	4.67,
	4.67,
	4.68,
	4.68,
	4.68,
	4.68,
	4.68,
	4.68,
	4.68,
	4.68,
	4.69,
	4.69,
	4.69,
	4.69,
	4.69,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.70,
	4.71,
	4.71,
	4.71,
	4.71,
	4.72,
	4.72,
	4.72,
	4.72,
	4.72,
	4.73,
	4.73,
	4.74,
	4.74,
	4.74,
	4.74,
	4.74,
	4.74,
	4.74,
	4.74,
	4.74,
	4.74,
	4.75,
	4.75,
	4.75,
	4.76,
	4.76,
	4.76,
	4.77,
	4.77,
	4.77,
	4.77,
	4.77,
	4.77,
	4.77,
	4.77,
	4.78,
	4.79,
	4.79,
	4.79,
	4.79,
	4.80,
	4.80,
	4.80,
	4.80,
	4.80,
	4.80,
	4.80,
	4.80,
	4.80,
	4.81,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.82,
	4.83,
	4.84,
	4.84,
	4.84,
	4.84,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.85,
	4.86,
	4.86,
	4.87,
	4.87,
	4.88,
	4.88,
	4.88,
	4.89,
	4.89,
	4.89,
	4.89,
	4.89,
	4.89,
	4.89,
	4.89,
	4.89,
	4.89,
	4.90,
	4.91,
	4.91,
	4.91,
	4.92,
	4.92,
	4.92,
	4.92,
	4.92,
	4.92,
	4.92,
	4.93,
	4.93,
	4.94,
	4.94,
	4.94,
	4.94,
	4.94,
	4.94,
	4.95,
	4.95,
	4.95,
	4.96,
	4.96,
	4.96,
	4.96,
	4.96,
	4.96,
	4.96,
	4.96,
	4.96,
	4.96,
	4.96,
	4.96,
	4.97,
	4.97,
	4.97,
	4.98,
	4.99,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.00,
	5.01,
	5.01,
	5.01,
	5.01,
	5.01,
	5.02,
	5.02,
	5.02,
	5.02,
	5.02,
	5.03,
	5.03,
	5.03,
	5.04,
	5.04,
	5.04,
	5.04,
	5.05,
	5.05,
	5.05,
	5.05,
	5.05,
	5.05,
	5.05,
	5.05,
	5.06,
	5.06,
	5.06,
	5.06,
	5.07,
	5.07,
	5.07,
	5.07,
	5.08,
	5.08,
	5.08,
	5.08,
	5.08,
	5.09,
	5.09,
	5.09,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.10,
	5.11,
	5.11,
	5.11,
	5.11,
	5.12,
	5.12,
	5.12,
	5.12,
	5.12,
	5.12,
	5.13,
	5.13,
	5.14,
	5.14,
	5.14,
	5.14,
	5.14,
	5.15,
	5.15,
	5.15,
	5.15,
	5.15,
	5.15,
	5.15,
	5.15,
	5.15,
	5.15,
	5.15,
	5.16,
	5.17,
	5.17,
	5.17,
	5.18,
	5.18,
	5.18,
	5.18,
	5.18,
	5.19,
	5.19,
	5.19,
	5.19,
	5.19,
	5.19,
	5.19,
	5.20,
	5.20,
	5.20,
	5.20,
	5.21,
	5.21,
	5.21,
	5.21,
	5.21,
	5.21,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.22,
	5.23,
	5.23,
	5.23,
	5.23,
	5.24,
	5.24,
	5.24,
	5.24,
	5.24,
	5.24,
	5.24,
	5.24,
	5.24,
	5.24,
	5.25,
	5.25,
	5.26,
	5.26,
	5.26,
	5.27,
	5.27,
	5.27,
	5.28,
	5.28,
	5.28,
	5.28,
	5.28,
	5.28,
	5.29,
	5.29,
	5.30,
	5.30,
	5.30,
	5.30,
	5.30,
	5.30,
	5.30,
	5.30,
	5.30,
	5.30,
	5.31,
	5.31,
	5.32,
	5.32,
	5.32,
	5.32,
	5.33,
	5.33,
	5.33,
	5.33,
	5.34,
	5.34,
	5.34,
	5.34,
	5.34,
	5.34,
	5.35,
	5.35,
	5.35,
	5.35,
	5.36,
	5.36,
	5.36,
	5.36,
	5.37,
	5.37,
	5.37,
	5.37,
	5.38,
	5.38,
	5.38,
	5.38,
	5.38,
	5.39,
	5.39,
	5.39,
	5.39,
	5.39,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.40,
	5.41,
	5.41,
	5.41,
	5.42,
	5.42,
	5.42,
	5.42,
	5.43,
	5.43,
	5.43,
	5.44,
	5.44,
	5.44,
	5.44,
	5.44,
	5.44,
	5.44,
	5.45,
	5.45,
	5.46,
	5.46,
	5.47,
	5.47,
	5.47,
	5.47,
	5.47,
	5.47,
	5.47,
	5.47,
	5.47,
	5.48,
	5.48,
	5.48,
	5.48,
	5.48,
	5.48,
	5.48,
	5.48,
	5.48,
	5.49,
	5.49,
	5.49,
	5.49,
	5.49,
	5.49,
	5.49,
	5.49,
	5.49,
	5.49,
	5.49,
	5.50,
	5.51,
	5.51,
	5.51,
	5.51,
	5.51,
	5.52,
	5.52,
	5.52,
	5.52,
	5.52,
	5.52,
	5.52,
	5.52,
	5.52,
	5.52,
	5.54,
	5.54,
	5.54,
	5.54,
	5.54,
	5.54,
	5.54,
	5.54,
	5.54,
	5.54,
	5.55,
	5.55,
	5.55,
	5.55,
	5.55,
	5.56,
	5.57,
	5.57,
	5.57,
	5.57,
	5.57,
	5.57,
	5.57,
	5.58,
	5.58,
	5.59,
	5.59,
	5.59,
	5.59,
	5.59,
	5.59,
	5.59,
	5.60,
	5.60,
	5.60,
	5.60,
	5.60,
	5.60,
	5.61,
	5.62,
	5.62,
	5.62,
	5.62,
	5.62,
	5.62,
	5.62,
	5.62,
	5.62,
	5.62,
	5.62,
	5.62,
	5.63,
	5.63,
	5.63,
	5.64,
	5.64,
	5.64,
	5.64,
	5.64,
	5.64,
	5.64,
	5.64,
	5.64,
	5.64,
	5.65,
	5.66,
	5.66,
	5.66,
	5.66,
	5.66,
	5.66,
	5.66,
	5.66,
	5.67,
	5.67,
	5.67,
	5.67,
	5.68,
	5.68,
	5.68,
	5.68,
	5.68,
	5.68,
	5.68,
	5.68,
	5.68,
	5.68,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.70,
	5.71,
	5.72,
	5.72,
	5.72,
	5.72,
	5.72,
	5.72,
	5.73,
	5.73,
	5.73,
	5.74,
	5.74,
	5.74,
	5.74,
	5.74,
	5.74,
	5.74,
	5.74,
	5.74,
	5.74,
	5.74,
	5.75,
	5.75,
	5.76,
	5.76,
	5.76,
	5.76,
	5.76,
	5.77,
	5.77,
	5.77,
	5.77,
	5.77,
	5.77,
	5.77,
	5.77,
	5.77,
	5.77,
	5.78,
	5.79,
	5.79,
	5.79,
	5.80,
	5.80,
	5.80,
	5.80,
	5.80,
	5.81,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.82,
	5.83,
	5.84,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.85,
	5.86,
	5.86,
	5.87,
	5.88,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.89,
	5.90,
	5.90,
	5.90,
	5.91,
	5.91,
	5.91,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.92,
	5.93,
	5.94,
	5.94,
	5.94,
	5.95,
	5.96,
	5.96,
	5.96,
	5.96,
	5.96,
	5.96,
	5.96,
	5.96,
	5.96,
	5.96,
	5.97,
	5.97,
	5.97,
	5.98,
	5.99,
	5.99,
	5.99,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.00,
	6.01,
	6.01,
	6.01,
	6.01,
	6.02,
	6.02,
	6.02,
	6.02,
	6.02,
	6.02,
	6.02,
	6.03,
	6.03,
	6.03,
	6.03,
	6.03,
	6.03,
	6.03,
	6.04,
	6.04,
	6.04,
	6.05,
	6.05,
	6.05,
	6.05,
	6.05,
	6.05,
	6.05,
	6.05,
	6.05,
	6.06,
	6.06,
	6.06,
	6.06,
	6.06,
	6.06,
	6.06,
	6.06,
	6.07,
	6.07,
	6.07,
	6.07,
	6.07,
	6.07,
	6.07,
	6.08,
	6.08,
	6.08,
	6.08,
	6.08,
	6.08,
	6.08,
	6.09,
	6.09,
	6.09,
	6.09,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.10,
	6.11,
	6.11,
	6.11,
	6.11,
	6.11,
	6.11,
	6.11,
	6.12,
	6.12,
	6.12,
	6.12,
	6.12,
	6.12,
	6.12,
	6.12,
	6.12,
	6.14,
	6.14,
	6.14,
	6.15,
	6.15,
	6.15,
	6.15,
	6.16,
	6.16,
	6.16,
	6.16,
	6.17,
	6.17,
	6.17,
	6.17,
	6.17,
	6.17,
	6.17,
	6.17,
	6.17,
	6.17,
	6.17,
	6.18,
	6.18,
	6.18,
	6.18,
	6.19,
	6.19,
	6.19,
	6.19,
	6.19,
	6.19,
	6.20,
	6.20,
	6.20,
	6.20,
	6.20,
	6.20,
	6.20,
	6.21,
	6.21,
	6.21,
	6.21,
	6.22,
	6.22,
	6.22,
	6.22,
	6.22,
	6.22,
	6.22,
	6.22,
	6.22,
	6.22,
	6.22,
	6.23,
	6.23,
	6.23,
	6.23,
	6.24,
	6.24,
	6.25,
	6.25,
	6.25,
	6.25,
	6.25,
	6.25,
	6.26,
	6.26,
	6.26,
	6.26,
	6.26,
	6.26,
	6.27,
	6.27,
	6.27,
	6.27,
	6.28,
	6.28,
	6.28,
	6.28,
	6.28,
	6.29,
	6.29,
	6.29,
	6.29,
	6.29,
	6.29,
	6.29,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.30,
	6.31,
	6.31,
	6.31,
	6.31,
	6.31,
	6.31,
	6.31,
	6.31,
	6.31,
	6.32,
	6.32,
	6.32,
	6.32,
	6.32,
	6.32,
	6.33,
	6.33,
	6.33,
	6.34,
	6.34,
	6.34,
	6.34,
	6.34,
	6.34,
	6.35,
	6.35,
	6.35,
	6.35,
	6.35,
	6.35,
	6.35,
	6.35,
	6.35,
	6.36,
	6.36,
	6.36,
	6.36,
	6.36,
	6.36,
	6.37,
	6.37,
	6.37,
	6.37,
	6.37,
	6.37,
	6.37,
	6.38,
	6.38,
	6.39,
	6.39,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.40,
	6.41,
	6.41,
	6.41,
	6.41,
	6.41,
	6.41,
	6.41,
	6.41,
	6.42,
	6.42,
	6.42,
	6.43,
	6.43,
	6.43,
	6.43,
	6.43,
	6.43,
	6.43,
	6.44,
	6.44,
	6.44,
	6.44,
	6.44,
	6.44,
	6.44,
	6.45,
	6.45,
	6.46,
	6.46,
	6.46,
	6.46,
	6.46,
	6.46,
	6.46,
	6.46,
	6.46,
	6.47,
	6.47,
	6.47,
	6.47,
	6.47,
	6.48,
	6.48,
	6.48,
	6.48,
	6.48,
	6.48,
	6.48,
	6.48,
	6.49,
	6.49,
	6.49,
	6.49,
	6.49,
	6.49,
	6.50,
	6.50,
	6.50,
	6.50,
	6.50,
	6.50,
	6.51,
	6.51,
	6.51,
	6.51,
	6.51,
	6.51,
	6.51,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.52,
	6.53,
	6.53,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.54,
	6.55,
	6.55,
	6.55,
	6.55,
	6.55,
	6.55,
	6.55,
	6.55,
	6.55,
	6.55,
	6.55,
	6.55,
	6.56,
	6.56,
	6.56,
	6.57,
	6.57,
	6.57,
	6.57,
	6.57,
	6.57,
	6.57,
	6.57,
	6.57,
	6.57,
	6.58,
	6.58,
	6.58,
	6.58,
	6.59,
	6.59,
	6.59,
	6.59,
	6.59,
	6.59,
	6.59,
	6.59,
	6.59,
	6.59,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.60,
	6.62,
	6.62,
	6.62,
	6.62,
	6.62,
	6.62,
	6.62,
	6.62,
	6.62,
	6.62,
	6.62,
	6.62,
	6.63,
	6.63,
	6.63,
	6.63,
	6.64,
	6.64,
	6.64,
	6.64,
	6.64,
	6.64,
	6.64,
	6.65,
	6.65,
	6.66,
	6.66,
	6.66,
	6.66,
	6.66,
	6.66,
	6.66,
	6.66,
	6.66,
	6.66,
	6.66,
	6.66,
	6.67,
	6.68,
	6.68,
	6.68,
	6.68,
	6.68,
	6.68,
	6.68,
	6.68,
	6.68,
	6.68,
	6.68,
	6.69,
	6.69,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.70,
	6.71,
	6.71,
	6.72,
	6.72,
	6.72,
	6.72,
	6.72,
	6.72,
	6.72,
	6.72,
	6.72,
	6.72,
	6.72,
	6.72,
	6.73,
	6.73,
	6.73,
	6.73,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.74,
	6.75,
	6.75,
	6.76,
	6.77,
	6.77,
	6.77,
	6.77,
	6.77,
	6.77,
	6.77,
	6.77,
	6.77,
	6.77,
	6.78,
	6.79,
	6.79,
	6.79,
	6.80,
	6.80,
	6.80,
	6.80,
	6.80,
	6.80,
	6.80,
	6.80,
	6.81,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.82,
	6.83,
	6.84,
	6.84,
	6.84,
	6.84,
	6.85,
	6.85,
	6.85,
	6.85,
	6.85,
	6.85,
	6.85,
	6.85,
	6.85,
	6.85,
	6.85,
	6.85,
	6.86,
	6.86,
	6.87,
	6.87,
	6.87,
	6.87,
	6.87,
	6.88,
	6.88,
	6.88,
	6.88,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.89,
	6.90,
	6.90,
	6.90,
	6.90,
	6.91,
	6.91,
	6.92,
	6.92,
	6.92,
	6.92,
	6.92,
	6.92,
	6.92,
	6.92,
	6.92,
	6.92,
	6.92,
	6.93,
	6.94,
	6.95,
	6.95,
	6.95,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.96,
	6.97,
	6.98,
	6.98,
	6.98,
	6.98,
	6.99,
	6.99,
	6.99,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.00,
	7.01,
	7.02,
	7.02,
	7.02,
	7.02,
	7.02,
	7.02,
	7.03,
	7.03,
	7.03,
	7.04,
	7.04,
	7.05,
	7.05,
	7.05,
	7.05,
	7.05,
	7.05,
	7.05,
	7.05,
	7.05,
	7.05,
	7.06,
	7.06,
	7.06,
	7.07,
	7.07,
	7.07,
	7.07,
	7.07,
	7.07,
	7.07,
	7.08,
	7.08,
	7.08,
	7.08,
	7.09,
	7.09,
	7.09,
	7.09,
	7.09,
	7.09,
	7.10,
	7.10,
	7.10,
	7.10,
	7.10,
	7.10,
	7.10,
	7.10,
	7.10,
	7.10,
	7.10,
	7.10,
	7.11,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.12,
	7.13,
	7.13,
	7.13,
	7.13,
	7.13,
	7.13,
	7.14,
	7.14,
	7.14,
	7.14,
	7.14,
	7.15,
	7.15,
	7.15,
	7.15,
	7.15,
	7.15,
	7.15,
	7.15,
	7.15,
	7.15,
	7.15,
	7.15,
	7.16,
	7.16,
	7.16,
	7.16,
	7.17,
	7.17,
	7.17,
	7.17,
	7.17,
	7.17,
	7.17,
	7.17,
	7.18,
	7.18,
	7.19,
	7.19,
	7.19,
	7.19,
	7.19,
	7.19,
	7.19,
	7.19,
	7.19,
	7.19,
	7.20,
	7.20,
	7.20,
	7.20,
	7.20,
	7.20,
	7.20,
	7.21,
	7.21,
	7.21,
	7.21,
	7.21,
	7.21,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.22,
	7.23,
	7.23,
	7.23,
	7.23,
	7.24,
	7.24,
	7.24,
	7.24,
	7.24,
	7.24,
	7.24,
	7.24,
	7.24,
	7.24,
	7.25,
	7.25,
	7.25,
	7.25,
	7.25,
	7.25,
	7.25,
	7.26,
	7.26,
	7.26,
	7.26,
	7.26,
	7.26,
	7.26,
	7.26,
	7.27,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.28,
	7.29,
	7.29,
	7.29,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.30,
	7.31,
	7.31,
	7.31,
	7.32,
	7.32,
	7.32,
	7.32,
	7.32,
	7.32,
	7.32,
	7.33,
	7.33,
	7.33,
	7.34,
	7.34,
	7.34,
	7.34,
	7.34,
	7.34,
	7.35,
	7.35,
	7.35,
	7.35,
	7.35,
	7.35,
	7.36,
	7.36,
	7.36,
	7.36,
	7.36,
	7.37,
	7.37,
	7.37,
	7.37,
	7.37,
	7.37,
	7.38,
	7.38,
	7.38,
	7.38,
	7.38,
	7.39,
	7.39,
	7.39,
	7.39,
	7.39,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.40,
	7.41,
	7.41,
	7.41,
	7.41,
	7.41,
	7.41,
	7.41,
	7.42,
	7.42,
	7.42,
	7.42,
	7.42,
	7.42,
	7.43,
	7.43,
	7.43,
	7.43,
	7.43,
	7.43,
	7.44,
	7.44,
	7.44,
	7.44,
	7.44,
	7.44,
	7.44,
	7.44,
	7.44,
	7.44,
	7.45,
	7.45,
	7.45,
	7.45,
	7.46,
	7.46,
	7.46,
	7.46,
	7.46,
	7.46,
	7.46,
	7.46,
	7.46,
	7.46,
	7.46,
	7.47,
	7.47,
	7.47,
	7.47,
	7.48,
	7.48,
	7.48,
	7.48,
	7.48,
	7.48,
	7.48,
	7.49,
	7.49,
	7.49,
	7.49,
	7.49,
	7.50,
	7.51,
	7.51,
	7.51,
	7.51,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.52,
	7.54,
	7.54,
	7.54,
	7.55,
	7.55,
	7.55,
	7.55,
	7.55,
	7.55,
	7.55,
	7.55,
	7.55,
	7.55,
	7.55,
	7.56,
	7.56,
	7.57,
	7.57,
	7.57,
	7.57,
	7.57,
	7.57,
	7.57,
	7.57,
	7.57,
	7.57,
	7.57,
	7.58,
	7.59,
	7.59,
	7.59,
	7.59,
	7.59,
	7.59,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.60,
	7.61,
	7.61,
	7.61,
	7.62,
	7.62,
	7.62,
	7.62,
	7.62,
	7.62,
	7.62,
	7.62,
	7.62,
	7.62,
	7.63,
	7.63,
	7.64,
	7.64,
	7.64,
	7.64,
	7.64,
	7.64,
	7.64,
	7.64,
	7.64,
	7.64,
	7.64,
	7.66,
	7.66,
	7.66,
	7.66,
	7.66,
	7.66,
	7.66,
	7.66,
	7.66,
	7.67,
	7.68,
	7.68,
	7.68,
	7.68,
	7.68,
	7.68,
	7.68,
	7.68,
	7.68,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.70,
	7.71,
	7.71,
	7.72,
	7.72,
	7.72,
	7.72,
	7.72,
	7.72,
	7.72,
	7.72,
	7.73,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.74,
	7.76,
	7.76,
	7.76,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.77,
	7.78,
	7.78,
	7.79,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.80,
	7.81,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.82,
	7.83,
	7.84,
	7.84,
	7.84,
	7.84,
	7.84,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.85,
	7.87,
	7.89,
	7.89,
	7.89,
	7.89,
	7.89,
	7.89,
	7.89,
	7.89,
	7.89,
	7.89,
	7.89,
	7.89,
	7.90,
	7.91,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.92,
	7.93,
	7.94,
	7.95,
	7.95,
	7.95,
	7.95,
	7.95,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.96,
	7.99,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.00,
	8.01,
	8.01,
	8.01,
	8.02,
	8.02,
	8.02,
	8.03,
	8.04,
	8.04,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.05,
	8.06,
	8.07,
	8.08,
	8.08,
	8.09,
	8.09,
	8.09,
	8.09,
	8.09,
	8.09,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.10,
	8.11,
	8.11,
	8.11,
	8.11,
	8.11,
	8.11,
	8.12,
	8.12,
	8.13,
	8.14,
	8.14,
	8.14,
	8.14,
	8.14,
	8.14,
	8.14,
	8.14,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.15,
	8.16,
	8.16,
	8.17,
	8.18,
	8.18,
	8.19,
	8.19,
	8.19,
	8.19,
	8.20,
	8.20,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.22,
	8.23,
	8.23,
	8.24,
	8.24,
	8.25,
	8.25,
	8.25,
	8.25,
	8.25,
	8.26,
	8.27,
	8.27,
	8.27,
	8.27,
	8.27,
	8.28,
	8.28,
	8.28,
	8.29,
	8.29,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.30,
	8.31,
	8.32,
	8.32,
	8.32,
	8.32,
	8.32,
	8.33,
	8.34,
	8.34,
	8.34,
	8.34,
	8.35,
	8.35,
	8.35,
	8.35,
	8.36,
	8.36,
	8.36,
	8.36,
	8.36,
	8.37,
	8.37,
	8.37,
	8.37,
	8.38,
	8.38,
	8.38,
	8.38,
	8.38,
	8.39,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.40,
	8.41,
	8.41,
	8.41,
	8.41,
	8.41,
	8.42,
	8.42,
	8.42,
	8.43,
	8.43,
	8.43,
	8.43,
	8.44,
	8.44,
	8.46,
	8.47,
	8.47,
	8.48,
	8.48,
	8.48,
	8.48,
	8.48,
	8.48,
	8.48,
	8.48,
	8.48,
	8.48,
	8.49,
	8.49,
	8.49,
	8.49,
	8.49,
	8.49,
	8.49,
	8.51,
	8.51,
	8.51,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.52,
	8.54,
	8.54,
	8.55,
	8.55,
	8.55,
	8.55,
	8.55,
	8.57,
	8.57,
	8.57,
	8.57,
	8.57,
	8.57,
	8.57,
	8.58,
	8.59,
	8.59,
	8.60,
	8.60,
	8.60,
	8.60,
	8.60,
	8.60,
	8.60,
	8.60,
	8.61,
	8.62,
	8.62,
	8.62,
	8.62,
	8.62,
	8.62,
	8.63,
	8.64,
	8.64,
	8.64,
	8.64,
	8.64,
	8.64,
	8.64,
	8.64,
	8.64,
	8.65,
	8.66,
	8.66,
	8.66,
	8.66,
	8.68,
	8.69,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.70,
	8.72,
	8.72,
	8.72,
	8.72,
	8.72,
	8.74,
	8.74,
	8.74,
	8.74,
	8.74,
	8.74,
	8.74,
	8.74,
	8.77,
	8.77,
	8.77,
	8.77,
	8.77,
	8.77,
	8.77,
	8.77,
	8.77,
	8.77,
	8.80,
	8.80,
	8.80,
	8.80,
	8.80,
	8.80,
	8.81,
	8.82,
	8.82,
	8.82,
	8.82,
	8.82,
	8.82,
	8.82,
	8.82,
	8.83,
	8.83,
	8.85,
	8.85,
	8.85,
	8.85,
	8.85,
	8.85,
	8.87,
	8.89,
	8.89,
	8.89,
	8.89,
	8.89,
	8.89,
	8.89,
	8.89,
	8.90,
	8.91,
	8.92,
	8.92,
	8.92,
	8.92,
	8.92,
	8.92,
	8.94,
	8.96,
	8.96,
	8.96,
	8.96,
	8.96,
	8.96,
	8.96,
	8.96,
	8.96,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.00,
	9.01,
	9.01,
	9.03,
	9.04,
	9.04,
	9.04,
	9.04,
	9.04,
	9.05,
	9.05,
	9.05,
	9.05,
	9.05,
	9.05,
	9.05,
	9.05,
	9.07,
	9.07,
	9.07,
	9.07,
	9.08,
	9.08,
	9.08,
	9.08,
	9.08,
	9.09,
	9.09,
	9.09,
	9.09,
	9.10,
	9.10,
	9.10,
	9.10,
	9.10,
	9.10,
	9.10,
	9.10,
	9.10,
	9.10,
	9.10,
	9.11,
	9.12,
	9.13,
	9.13,
	9.13,
	9.15,
	9.15,
	9.15,
	9.15,
	9.15,
	9.15,
	9.15,
	9.18,
	9.18,
	9.19,
	9.19,
	9.19,
	9.19,
	9.21,
	9.21,
	9.22,
	9.22,
	9.22,
	9.22,
	9.22,
	9.22,
	9.22,
	9.22,
	9.22,
	9.22,
	9.22,
	9.22,
	9.24,
	9.26,
	9.27,
	9.27,
	9.27,
	9.27,
	9.28,
	9.28,
	9.28,
	9.28,
	9.28,
	9.28,
	9.28,
	9.29,
	9.30,
	9.30,
	9.30,
	9.30,
	9.30,
	9.31,
	9.31,
	9.32,
	9.32,
	9.34,
	9.34,
	9.35,
	9.37,
	9.39,
	9.39,
	9.40,
	9.40,
	9.40,
	9.40,
	9.40,
	9.40,
	9.40,
	9.40,
	9.40,
	9.40,
	9.41,
	9.41,
	9.42,
	9.42,
	9.43,
	9.44,
	9.44,
	9.46,
	9.46,
	9.47,
	9.48,
	9.49,
	9.51,
	9.51,
	9.51,
	9.52,
	9.52,
	9.52,
	9.52,
	9.52,
	9.52,
	9.52,
	9.52,
	9.52,
	9.52,
	9.52,
	9.54,
	9.54,
	9.54,
	9.55,
	9.55,
	9.55,
	9.57,
	9.57,
	9.57,
	9.57,
	9.59,
	9.59,
	9.60,
	9.62,
	9.64,
	9.64,
	9.64,
	9.64,
	9.64,
	9.65,
	9.68,
	9.70,
	9.70,
	9.70,
	9.70,
	9.70,
	9.70,
	9.70,
	9.70,
	9.70,
	9.72,
	9.72,
	9.74,
	9.74,
	9.74,
	9.76,
	9.77,
	9.77,
	9.80,
	9.80,
	9.80,
	9.82,
	9.82,
	9.82,
	9.82,
	9.84,
	9.85,
	9.85,
	9.85,
	9.89,
	9.89,
	9.89,
	9.92,
	9.92,
	9.92,
	9.92,
	9.92,
	9.95,
	9.96,
	9.96,
	9.96,
	9.96,
	10.00,
	10.00,
	10.00,
	10.00,
	10.00,
	10.00,
	10.00,
	10.00,
	10.00,
	10.00,
	10.00,
	10.01,
	10.05,
	10.05,
	10.06,
	10.07,
	10.10,
	10.10,
	10.10,
	10.11,
	10.16,
	10.17,
	10.17,
	10.17,
	10.18,
	10.19,
	10.20,
	10.20,
	10.21,
	10.23,
	10.24,
	10.25,
	10.30,
	10.30,
	10.30,
	10.30,
	10.32,
	10.33,
	10.35,
	10.35,
	10.36,
	10.37,
	10.39,
	10.42,
	10.44,
	10.44,
	10.46,
	10.48,
	10.49,
	10.49,
	10.51,
	10.52,
	10.52,
	10.52,
	10.52,
	10.55,
	10.55,
	10.57,
	10.57,
	10.59,
	10.59,
	10.60,
	10.62,
	10.62,
	10.62,
	10.64,
	10.66,
	10.68,
	10.70,
	10.70,
	10.70,
	10.70,
	10.70,
	10.70,
	10.72,
	10.72,
	10.72,
	10.77,
	10.78,
	10.80,
	10.80,
	10.82,
	10.82,
	10.85,
	10.85,
	10.89,
	10.90,
	10.92,
	10.92,
	10.96,
	10.96,
	10.96,
	11.00,
	11.00,
	11.05,
	11.05,
	11.06,
	11.07,
	11.10,
	11.11,
	11.15,
	11.20,
	11.22,
	11.22,
	11.23,
	11.24,
	11.26,
	11.28,
	11.30,
	11.30,
	11.30,
	11.30,
	11.30,
	11.30,
	11.32,
	11.33,
	11.34,
	11.35,
	11.40,
	11.40,
	11.42,
	11.52,
	11.52,
	11.52,
	11.54,
	11.57,
	11.57,
	11.59,
	11.68,
	11.68,
	11.70,
	11.72,
	11.74,
	11.80,
	11.82,
	11.82,
	11.85,
	11.92,
}};
