using System;

namespace HalykBank
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Enter your name:");
            string name = Console.ReadLine();
            Console.WriteLine("Hello, {0}!", name);
            Console.WriteLine("Press any key to close");
            Console.ReadKey();
        }
    }
}
