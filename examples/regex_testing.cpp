#include <iostream>
#include <regex>
#include <unordered_map>

int main() {
  std::string main =
      " x[0] = 3. + 1.4923e-5 - 0.1234;\n jac[2] = -312.32 + 4312.32 - 312.322 "
      "* 0312.32 * 823.12e05 + "
      "3.135e-08;";
  std::match_results<std::string::const_iterator> res;
  std::string::const_iterator begin = main.cbegin(), end = main.cend();
  std::unordered_map<std::string, int> counts;
  while (std::regex_search(
      begin, end, res, std::basic_regex(" ([\\-]?[0-9\\.]+(e[\\-0-9]+)?)"))) {
    const auto &r = res[1];
    std::cout << r.length() << "  -  " << r.str() << "\n";
    begin = r.second;
    std::string match = r.str();
    if (match.size() < 3) {
      continue;  // don't bother with short numbers
    }
    if (counts.find(match) == counts.end()) {
      counts[match] = 0;
    }
    counts[match]++;
  }
  size_t const_id = 0;
  for (const auto &[value, count] : counts) {
    if (count > 0) {
      std::string varname = "c" + std::to_string(const_id++);
      size_t start = 0;
      std::basic_regex value_regex("[\\W;](" + value + ")[\\W;]");
      //   main = std::regex_replace(main, value_regex, varname);

      bool match_found = true;
      while (match_found) {
        begin = main.cbegin();
        end = main.cend();
        match_found = std::regex_search(begin, end, res, value_regex);
        if (match_found) {
          main = main.replace(res[1].first, res[1].second, varname);
        }
      }
      main = "static const Float " + varname + " = " + value + ";\n" + main;
    }
  }
  if (res.empty()) {
    std::cout << "No match\n";
  }
  std::cout << main << "\n";
  return EXIT_SUCCESS;
}