#pragma once

#include "neutron/config.hpp"
#include "neutron/engine.hpp"

namespace neutron {

int run_server(const Config & config, Engine & engine);

} // namespace neutron
