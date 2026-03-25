// Package main is the entry point for the icp-slam Viam module.
package main

import (
	"context"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/services/slam"
	"go.viam.com/utils"

	"github.com/viamrobotics/icp/viamslam"
)

func main() {
	utils.ContextualMain(mainWithArgs, module.NewLoggerFromArgs("icp-slam"))
}

func mainWithArgs(ctx context.Context, args []string, logger logging.Logger) error {
	icpModule, err := module.NewModuleFromArgs(ctx)
	if err != nil {
		return err
	}

	if err = icpModule.AddModelFromRegistry(ctx, slam.API, viamslam.Model); err != nil {
		return err
	}

	if err = icpModule.Start(ctx); err != nil {
		return err
	}
	defer icpModule.Close(ctx)

	<-ctx.Done()
	return nil
}
