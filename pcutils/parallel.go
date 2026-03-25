package pcutils

import (
	"context"

	goutils "go.viam.com/utils"
)

// NOTE: If two goroutines have access to the same ParallelWorkers object, it's possible to panic
// if one of them calls Wait() while another calls Do(). This class is intended to be used by a
// single goroutine, as a way to parallelize the work done within it.

// Example pattern:
//
// workers := NewParallelWorkers(10) // Spin up 10 worker goroutines
// for item := range items {
//
//	workers.Do(SomethingWith(item))
//
// }
// workers.Wait()
type ParallelWorkers struct {
	workers *goutils.StoppableWorkers
	channel chan func()
}

// NewParallelWorkers creates a pool of threadCount worker goroutines.
func NewParallelWorkers(threadCount int) ParallelWorkers {
	channel := make(chan func())
	result := ParallelWorkers{
		workers: goutils.NewBackgroundStoppableWorkers(),
		channel: channel,
	}

	for i := 0; i < threadCount; i++ {
		result.workers.Add(func(ctx context.Context) {
			for {
				select {
				case <-ctx.Done():
					return
				case f := <-channel:
					f()
				}
			}
		})
	}

	return result
}

// Do submits a function to be executed by one of the worker goroutines.
func (pw *ParallelWorkers) Do(f func()) {
	pw.channel <- f
}

// Wait stops all workers and waits for them to finish.
func (pw *ParallelWorkers) Wait() {
	pw.workers.Stop()
	close(pw.channel)
}
