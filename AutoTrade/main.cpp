#include <iostream>
#include <curl/curl.h>
#include <string>
#include <nlohmann/json.hpp>
#include <torch/script.h>
#include <vector>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <boost/asio.hpp>
#include <thread>
#include <chrono>
using json = nlohmann::json;
using websocketpp::connection_hdl;
typedef websocketpp::client<websocketpp::config::asio_client> client;

const std::string apipassword = "9BjmVz5OZF";
const std::string password = "9BjmVz5OZF";
const std::string port = "80";
const std::vector<std::string> symbols = {"7203"};
std::string token = "";
torch::jit::script::Module model;

std::vector<float> currentPrices(symbols.size(), 0.0);
std::vector<float> currentVolumes(symbols.size(), 0.0);
std::vector<float> previousVolumes(symbols.size(), 0.0);
std::vector<std::vector<float>> historicalPrices(symbols.size(), std::vector<float>(0));
std::vector<std::vector<float>> historicalPriceChanges(symbols.size(), std::vector<float>(0));
std::vector<std::vector<float>> historicalVolumes(symbols.size(), std::vector<float>(0));
std::vector<std::vector<float>> historicalVolumeChanges(symbols.size(), std::vector<float>(0));
std::vector<std::vector<float>> historicalMinimumVolalities(symbols.size(), std::vector<float>(0));
std::vector<std::string> priceRangeGroups(0);

void on_message(client *c, websocketpp::connection_hdl hdl, client::message_ptr msg)
{
    try
    {
        auto m = json::parse(msg->get_payload());
        std::string s = m["Symbol"];
        double p = m["CurrentPrice"];
        double v = m["TradingVolume"];

        auto it = std::find(symbols.begin(), symbols.end(), s);
        if (it != symbols.end())
        {
            int i = std::distance(symbols.begin(), it);
            currentPrices[i] = p;
            currentVolumes[i] = v;
        }
    }
    catch (const json::exception &e)
    {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
    }
}

void on_error(client *c, websocketpp::connection_hdl hdl)
{
    client::connection_ptr con = c->get_con_from_hdl(hdl);
    std::cerr << con->get_ec().message() << std::endl;
}

void on_close(client *c, websocketpp::connection_hdl hdl)
{
    std::cerr << "--- DISCONNECTED --- " << std::endl;
    while (true)
    {
        try
        {
            std::cerr << "Reconnecting..." << std::endl;
            websocket_thread();
            break;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Reconnection failed: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1)); // 再接続の間隔を調整
        }
    }
}

void on_open(client *c, websocketpp::connection_hdl hdl)
{
    std::cerr << "--- CONNECTED --- " << std::endl;
}

void start_websocket(client *c)
{
    std::cerr << "connecting" << std::endl;
    std::string uri = "ws://172.30.96.1:80/kabusapi/websocket";

    websocketpp::lib::error_code ec;
    client::connection_ptr con = c->get_connection(uri, ec);
    if (ec)
    {
        std::cerr << "Could not create connection because: " << ec.message() << std::endl;
        return;
    }

    c->connect(con);
}

void websocket_thread()
{
    client c;

    try
    {
        c.set_access_channels(websocketpp::log::alevel::all);
        c.clear_access_channels(websocketpp::log::alevel::frame_payload);

        c.init_asio();

        c.set_message_handler(websocketpp::lib::bind(&on_message, &c, websocketpp::lib::placeholders::_1, websocketpp::lib::placeholders::_2));
        c.set_open_handler(websocketpp::lib::bind(&on_open, &c, websocketpp::lib::placeholders::_1));
        c.set_close_handler(websocketpp::lib::bind(&on_close, &c, websocketpp::lib::placeholders::_1));
        c.set_fail_handler(websocketpp::lib::bind(&on_error, &c, websocketpp::lib::placeholders::_1));

        start_websocket(&c);

        c.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch (websocketpp::lib::error_code e)
    {
        std::cerr << e.message() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Other exception" << std::endl;
    }
}

size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s)
{
    s->append((char *)contents, size * nmemb);
    return size * nmemb;
}

json postRequest(const std::string &url, const std::string &data, struct curl_slist *headers)
{
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

        std::cout << "HTTP Status: " << response_code << std::endl;
        std::cout << "HTTP Response: " << readBuffer << std::endl;

        curl_easy_cleanup(curl);
        curl_global_cleanup();

        if (response_code == 200)
        {
            return json::parse(readBuffer);
        }
        else
        {
            throw std::runtime_error("Request failed with status code " + std::to_string(response_code));
        }
    }
    curl_global_cleanup();

    return json();
}

json getRequest(const std::string &url, struct curl_slist *headers)
{
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

        std::cout << "HTTP Status: " << response_code << std::endl;
        std::cout << "HTTP Response: " << readBuffer << std::endl;

        curl_easy_cleanup(curl);
        curl_global_cleanup();

        if (response_code == 200)
        {
            return json::parse(readBuffer);
        }
        else
        {
            throw std::runtime_error("Request failed with status code " + std::to_string(response_code));
        }
    }
    curl_global_cleanup();

    return json();
}

json putRequest(const std::string &url, const std::string &data, struct curl_slist *headers)
{
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT"); // Use PUT request
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

        std::cout << "HTTP Status: " << response_code << std::endl;
        std::cout << "HTTP Response: " << readBuffer << std::endl;

        curl_easy_cleanup(curl);
        curl_global_cleanup();

        if (response_code == 200)
        {
            return json::parse(readBuffer);
        }
        else
        {
            throw std::runtime_error("Request failed with status code " + std::to_string(response_code));
        }
    }
    curl_global_cleanup();

    return json();
}

json getToken()
{
    std::string url = "http://172.30.96.1:" + port + "/kabusapi/token";
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    json obj = {{"APIPassword", apipassword}};
    try
    {
        json response = postRequest(url, obj.dump(), headers);
        std::cout << "Response JSON: " << response.dump(4) << std::endl;
        return response;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return json();
    }
}

json getMargin()
{
    std::string url = "http://172.30.96.1:" + port + "/wallet/margin";
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, ("X-API-KEY: " + token).c_str());
    try
    {
        json response = getRequest(url, headers);
        std::cout << "Response JSON: " << response.dump(4) << std::endl;
        return response;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return json();
    }
}

json sendOrder(std::string symbol, int side, float price, int qty)
{
    std::string url = "http://172.30.96.1:" + port + "/kabusapi/sendorder";
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, ("X-API-KEY: " + token).c_str());
    json obj;
    if (side == 1)
    {
        obj = {{"Password", password},
               {"Symbol", symbol},
               {"Exchange", 1},
               {"SecurityType", 1},
               {"Side", side},
               {"CashMargin", 3},
               {"MarginTradeType", 3},
               {"DelivType", 0},
               {"AccountType", 2},
               {"Qty", qty},
               {"ClosePositionOrder", 0},
               {"FrontOrderType", 20},
               {"Price", price},
               {"ExpireDay", 0}};
    }
    else if (side == 2)
    {
        obj = {{"Password", password},
               {"Symbol", symbol},
               {"Exchange", 1},
               {"SecurityType", 1},
               {"Side", side},
               {"CashMargin", 2},
               {"MarginTradeType", 3},
               {"DelivType", 0},
               {"AccountType", 2},
               {"Qty", qty},
               {"FrontOrderType", 20},
               {"Price", price},
               {"ExpireDay", 0}};
    }
    try
    {
        json response = postRequest(url, obj.dump(), headers);
        std::cout << "Response JSON: " << response.dump(4) << std::endl;
        return response;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return json();
    }
}

json getOrder(std::string id)
{
    std::string url = "http://172.30.96.1:" + port + "/kabusapi/orders?product=0&id" + id;
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, ("X-API-KEY: " + token).c_str());
    try
    {
        json response = getRequest(url, headers);
        std::cout << "Response JSON: " << response.dump(4) << std::endl;
        return response;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return json();
    }
}

json cancelOrder(std::string id)
{
    std::string url = "http://172.30.96.1:" + port + "/kabusapi/cancelorder";
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, ("X-API-KEY: " + token).c_str());
    json obj = {{"OrderID", id},
                {"Password", password}};
    try
    {
        json response = putRequest(url, obj.dump(), headers);
        std::cout << "Response JSON: " << response.dump(4) << std::endl;
        return response;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return json();
    }
}

json priceRangeGroup(std::string symbol)
{
    std::string url = "http://172.30.96.1:" + port + "/kabusapi/symbol/" + symbol + "@1?addinfo=false";
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, ("X-API-KEY: " + token).c_str());
    try
    {
        json response = getRequest(url, headers);
        std::cout << "Response JSON: " << response.dump(4) << std::endl;
        return response;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return json();
    }
}

json regi()
{
    std::string url = "http://172.30.96.1:" + port + "/kabusapi/unregister/all";
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, ("X-API-KEY: " + token).c_str());
    putRequest(url, "", headers);

    url = "http://172.30.96.1:" + port + "/kabusapi/register";
    json obj;
    for (const auto &s : symbols)
    {
        json symbol_obj;
        symbol_obj["Symbol"] = s;
        symbol_obj["Exchange"] = 1;
        obj["Symbols"].push_back(symbol_obj);
    }
    try
    {
        json response = putRequest(url, obj.dump(), headers);
        std::cout << "Response JSON: " << response.dump(4) << std::endl;
        return response;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return json();
    }
}

float minumumVolality(float price, std::string priceRangeGroup)
{
    float mv = 0.0;
    if (priceRangeGroup == "10003")
    {
        if (price <= 1000)
        {
            mv = 0.1 / price * 100;
        }
        else if (1000 < price && price <= 3000)
        {
            mv = 0.5 / price * 100;
        }
        else if (3000 < price && price <= 5000)
        {
            mv = 1 / price * 100;
        }
        else if (5000 < price && price <= 10000)
        {
            mv = 1 / price * 100;
        }
        else if (10000 < price && price <= 30000)
        {
            mv = 5 / price * 100;
        }
        else if (30000 < price && price <= 50000)
        {
            mv = 10 / price * 100;
        }
        else if (50000 < price)
        {
            mv = 10 / price * 100;
        }
    }
    else
    {
        if (price <= 1000)
        {
            mv = 1 / price * 100;
        }
        else if (1000 < price && price <= 3000)
        {
            mv = 1 / price * 100;
        }
        else if (3000 < price && price <= 5000)
        {
            mv = 5 / price * 100;
        }
        else if (5000 < price && price <= 10000)
        {
            mv = 10 / price * 100;
        }
        else if (10000 < price && price <= 30000)
        {
            mv = 10 / price * 100;
        }
        else if (30000 < price && price <= 50000)
        {
            mv = 50 / price * 100;
        }
        else if (50000 < price)
        {
            mv = 100 / price * 100;
        }
    }

    return mv;
}

torch::Tensor judge(torch::Tensor data)
{
    return torch::softmax(model.forward({data}).toTensor(), 1);
}

int main()
{
    model = torch::jit::load("model.pth");

    std::string port = "80";

    std::string obj = "{\"APIPassword\":\"9BjmVz5OZF\"}";
    json obj1 = {{"APIPassword", "9BjmVz5OZF"}};
    std::string s_obj1 = obj1.dump();
    std::string url = "http://172.30.96.1:" + port + "/kabusapi/token";
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    try
    {
        json response = postRequest(url, s_obj1, headers);
        std::cout << "Response JSON: " << response.dump(4) << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    getToken();
    for (const auto &s : symbols)
    {
        priceRangeGroups.push_back(s);
    }

    return 0;
}
